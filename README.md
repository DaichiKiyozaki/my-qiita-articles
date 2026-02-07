# my-qiita-articles

Qiitaの記事を管理・GitHubActionsで自動同期するリポジトリ
- 参考：[QiitaとGitHubを連携して記事を投稿してみた](https://zenn.dev/kuuki/articles/qiita-article-managed-by-github)
- 画像も相対パス -> publicなurlに自動変換する

## 使い方

- 記事の作成
``` bash
npx qiita new {my_article_name}
```

- プレビュー
``` bash
npx qiita preview
```

- 記事を投稿・更新
    - ローカルで任意の記事を更新し、GitHubリポジトリにpush