---
title: YOLO-segをファインチューニングして「人検知＋向き推定」を1モデルに統合する
tags:
  - Python
  - 画像認識
  - YOLO
  - segmentation
private: false
updated_at: '2026-02-08T19:13:54+09:00'
id: 0d6c524522faeebb336c
organization_url_name: null
slide: false
ignorePublish: false
---

今回はYOLO-Segをファインチューニングすることにより、歩行者の「人検知」と「向き推定」を行える画像認識モデルを作成してみました。

※ここでいう向きは、一般的な細かい角度推定ではなく2値です。

* 同方向：背中向き。カメラから遠ざかる向き
* 逆方向：正面向き。カメラに近づく向き


ソースコードは[こちら](https://github.com/DaichiKiyozaki/seg-person-dir)です。

## 背景

研究要件として、歩行者のセグメンテーションと、向き推定（同方向/逆方向）が必要でした。

以前までは以下の2モデル構成で処理を行っていました。

* YOLO-seg：人のインスタンスセグメンテーション（人検知）
* MEBOW：各人物について体の向き（前後）を推定

つまり、検出モデル（YOLO-seg）と向き推定モデル（MEBOW）の2モデル構成です。

ただ、この構成は運用面で気になる点がありました。

* モデルが2つになり、依存関係と運用コストが増える
* MEBOWは人物ごとに推論するので、人が多いと処理が重くなる

今回本当に欲しい情報は、「歩行者がカメラに対して同方向か逆方向か」だけです。

それなら、YOLO-seg自体をファインチューニングして「向き」もクラスとして学習させればよいのでは、という方針にしました。
うまくいけば、人セグメンテーションと向き推定が1モデルで済みます。

## YOLO / YOLO-seg とは

YOLOは、画像を1回処理して複数物体を同時に検出するワンステージ型の検出モデルです。
YOLO-segは、BBoxだけでなくピクセル単位のマスクも出すインスタンスセグメンテーションです。

* Ultralytics YOLO Docs：https://docs.ultralytics.com/
* Instance Segmentation：https://docs.ultralytics.com/tasks/segment/

## ベースモデル（YOLO26s-seg）

今回は `YOLO26s-seg` をベースにしました。

特徴としては、ざっくり以下です。

* YOLO-seg系なので、BBoxに加えて**インスタンスマスク**を直接出せる
* `s` はサイズ感的に**速度と精度のバランス**が取りやすい
* モデルが大きすぎないので、まずは学習・推論を回して試行錯誤しやすい

また、Ultralyticsの YOLO26 はエッジ/低電力デバイスを意識して設計が整理されており、ドキュメント上も「エンドツーエンド（NMS不要）」「簡素化された推論」などが特徴として説明されています。

* https://docs.ultralytics.com/models/yolo26/

## ファインチューニングとは

ファインチューニングは、COCOなどで学習済みの重みを出発点にして、自分のデータ・自分のクラス定義に合わせて追加学習することです（いわゆる転移学習）。

こういうときにやります。

* COCOに存在しない概念を扱いたい（例：同方向/逆方向歩行者）
* 特定環境（屋内・監視カメラ・ロボット視点）に最適化したい
* 少ないデータでも学習を安定させたい（ゼロからより収束しやすい）

今回やりたいのは「同方向歩行者/逆方向歩行者」という独自ラベルなので、ファインチューニングで吸収させます。

### 独自クラス定義について

Ultralytics系はデータセットYAML側に `names:` を書けばクラス定義できます。さらに、データセット側の `nc`（クラス数）がモデル設定と違っても、データセットYAMLに合わせて自動で上書きされる旨がガイドに明記されています。

そのため、「COCO事前学習モデルを読み込んで、独自に定義した2クラスへ」という流れが可能です。

* https://docs.ultralytics.com/guides/model-yaml-config/

## COCOデータセットを使う理由

COCOのpersonには、セグメンテーションマスクと17点のキーポイントが入っています。
このキーポイントを使うと、「全身がある程度写っているか」を機械的にチェックできて便利です。

また、COCOはカテゴリで `person` を指定して抽出でき、さらにキーポイントで色々フィルタリングできるので、今回の要件（歩行者をセグメントしつつ前後の向きラベルを作る）に合致しています。

* COCO Dataset：https://cocodataset.org/
* COCO API（pycocotools）：https://github.com/cocodataset/cocoapi

## COCOデータの加工方針

今回の流れは以下です。

1. COCOからpersonを含む画像を抽出
2. キーポイントでノイズをフィルタ
3. MEBOWで向きを推定して2値ラベルを付与
4. 2クラスのYOLO-segとして学習

### ① person画像の抽出（COCO API）

COCO APIを使って、personを含む画像IDだけを抽出します。

* `getCatIds(catNms=['person'])`
* `getImgIds(catIds=...)`
* `getAnnIds(...)` / `loadAnns(...)`

最初からperson画像に絞れるので、後処理がだいぶ楽になります。


### ② ノイズ画像のフィルタリング（キーポイント利用）

COCOには、

* 人が小さすぎる
* 手だけ・腕だけ映っている
* 歩行者として不適切

なデータも混ざります。

そこで、**キーポイントを使ったフィルタリング**を行います。

（そのままでもある程度上手く学習できますが、これらのデータは向き推定にはノイズになるため、今回は精度向上のために除外します）

#### フィルタリング条件（例）

以下を満たす歩行者が一人も存在しない場合、その画像を学習データから除外します。

* keypoints の visibility が `v ≥ 1`
  * COCOのkeypointsは各点が `(x, y, v)` を持ち、`v` は次のフラグです
    * `v = 0`：そのキーポイントは未ラベル（座標なし）
    * `v = 1`：ラベルはあるが不可視（遮蔽などで見えていない）
    * `v = 2`：ラベルがあり可視
* 次のキーポイントが映っている（= ある程度全身が写っている目安）
  * `5, 6, 13, 14`
  * [キーポイント一覧表](https://qiita.com/kHz/items/8c06d0cb620f268f4b3e#keypoints-1)
    * 5, 6：左右の肩（left/right shoulder）
    * 13, 14：左右の膝（left/right knee）

狙いは「上半身と下半身の主要点が写っている個体がいる画像」を残して、歩行者として使えるデータの割合を増やすことです。

### ③ 任意数のデータを抽出（train/val/test）

* 学習用（train）：20000枚
* 検証用（val）：1482枚
* テスト用(test)：3000枚

今回はフィルタリングを通った画像のうち、上記の枚数を抽出します。

注意点：train/val/testで画像IDが重複しないように分割します。

### ④ クラス定義（2クラス）

今回のクラスは「歩行者がカメラに対してどちらを向いているか」で定義します。

* 0: 同方向（背中向き。カメラから遠ざかる向き）
* 1: 逆方向（正面向き。カメラに近づく向き）

ここでいう同方向/逆方向は、カメラ視点に対して同じ向きか逆向きか、という意味です。


## MEBOWとは

### MEBOWの役割

MEBOWは、

* 人のキーポイント情報などを手がかりに
* **身体の向き**を推定するモデル

です。

* 単眼RGB画像
* COCO系データを拡張した大規模データで学習

### 今回の立ち位置

* **学習データ作成用**（教師ラベル生成）
* 推論時には使わない
* 公式リポジトリで配布されている **Trained HBOE model** を使用しました
  * https://github.com/ChenyanWu/MEBOW?tab=readme-ov-file

MEBOWは教師ラベル生成器としてのみ使用します。


## 向きラベルの付与

流れはシンプルです。

1. MEBOWで各人物の向き角度 $\theta$ を推定する（$0\sim 360^\circ$）
2. 角度をルールで2値化してクラスを決める
3. YOLO-seg用アノテーションにクラスとして反映する

### 同方向/逆方向の判定条件

今回の実装では、MEBOWの出力角度 $\theta$ が「背中向き」を $0^\circ$ とする向きに対応する前提で、次の閾値にしました。

* 同方向（クラス0）：$\theta \le 45^\circ$ または $\theta \ge 315^\circ$
* 逆方向（クラス1）：それ以外

横向きは境界が曖昧になりやすいので、この閾値で割り切っています。

これにより、

* **人マスク**
* **向きクラス**

を同時に持つデータセットが完成します。

## YOLO26s-segのファインチューニング

`YOLO26s-seg` をベースにファインチューニングし、学習後は1回の推論で次を同時に出せるようになります。

* 人のセグメンテーション
* 同方向/逆方向の判定

今回使用した学習コマンドは以下です。

```bash
yolo segment train data=data/dataset_frontback_yoloseg/data.yaml model=yolo26s-seg.pt imgsz=512 epochs=50 batch=12 project=./runs/segment name=frontback
```


## 学習結果

### 学習結果グラフ

以下はUltralyticsの学習ログ出力です。最後の方にlossが上がっている箇所はありますが、全体としては学習が進んでいる様子が分かります。

![yolo-ft-results](https://raw.githubusercontent.com/DaichiKiyozaki/my-qiita-articles/main/images/yolo_ft_results.png)

### 推論例

人のマスクに加えて、各インスタンスが「同方向（青）/ 逆方向（赤）」に分類出来ていることが分かります。

![pred_ex](https://raw.githubusercontent.com/DaichiKiyozaki/my-qiita-articles/main/images/yolo_pred_ex.jpg)

※推論例に使った写真は [Unsplash](https://unsplash.com/ja/%E5%86%99%E7%9C%9F/%E6%98%BC%E9%96%93%E6%AD%A9%E8%A1%8C%E8%80%85%E5%B0%82%E7%94%A8%E9%81%93%E8%B7%AF%E3%82%92%E6%AD%A9%E3%81%8F%E7%B7%91%E8%89%B2%E3%81%AE%E3%82%B8%E3%83%A3%E3%82%B1%E3%83%83%E3%83%88%E3%82%92%E7%9D%80%E3%81%9F%E5%A5%B3%E6%80%A7-pzMP-RGJ7mY) からダウンロードしたものです


## まとめ

* YOLO-segをファインチューニングして、用途に合わせたモデルを獲得出来ました。
* COCOはpersonにキーポイントが付いているので、機械的なフィルタがしやすく、学習データのフィルタリングがやりやすかったです。
* YOLOのファインチューニングも初めて行ったのですが、意外と上手くいって良かったです！


## 参考リンク

* [Ultralytics YOLO Docs](https://docs.ultralytics.com/)
* [Instance Segmentation](https://docs.ultralytics.com/tasks/segment/)
* [COCO Dataset](https://cocodataset.org/)
* [pycocotools（COCO API）](https://github.com/cocodataset/cocoapi)
* [MEBOW](https://arxiv.org/abs/2203.08651)
