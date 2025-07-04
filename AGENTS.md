GPT4.1とANTsPyNetを用いたMRI読影支援MVP開発指示書
1. 全体目標と構成概要
目的: 本MVPは、OpenAIのGPT4.1（画像入力対応のVision機能）と医療画像解析ライブラリANTsPyNetを組み合わせ、ローカルPC上で動作するMRI読影支援ツールを構築することです。ユーザー（放射線技師や医師）がMRI画像をアップロードすると、バックエンドでANTsPyNetが専門的な画像解析（脳抽出や病変候補領域のセグメンテーションなど）を行い、続いてGPT4.1が画像と解析結果をもとに所見の要約や診断補助情報を生成します。この結果をStreamlitベースのインタラクティブなWebインターフェース上で可視化し、ユーザーに提供します。クラウドサービス（Google CloudやGemini）は使用せず、すべてローカル環境で完結する構成です。これにより、ネットワーク遅延や機密データの外部送信リスクを抑えつつ、迅速なプロトタイピングを実現します。
データフロー概要: ユーザー操作から結果表示までの一連の流れは以下のとおりです：
1. ユーザーがStreamlitアプリ上でMRI画像ファイルをアップロードします。
2. バックエンドでアップロード画像を受け取り、ANTsPyNetを用いた画像解析モジュールで脳抽出（スカルストリッピング）や病変候補のセグメンテーション処理を実行します。。ANTsPyNetは医療画像のセグメンテーションにおける最先端のオープンソースツールキットであり、高品質な脳抽出モデル（事前学習済みモデル）を提供しています。この処理で元画像に対する確率マップ（病変の可能性を示すマップ）やバイナリマスクが得られます。
3. 続いてレポート生成モジュールが、OpenAIのGPT4.1 APIを呼び出します。GPT4.1のマルチモーダル（Vision）機能により、元のMRI画像とANTsPyNet生成マスクを入力として与え、あらかじめ設計したプロンプトに基づいて所見のテキスト要約とメタデータを生成します。モデルには「JSON形式で出力する」指示を与えており、信頼度スコアや位置情報などを含む構造化データ（JSONオブジェクト）を応答として受け取ります。
4. StreamlitアプリはGPTからのJSON応答を解析し、結果表示フェーズでユーザーにフィードバックを提供します。具体的には、アップロードされたMRIのオリジナル画像にセグメンテーションマスクを重ね合わせたビュー（脳抽出の結果や病変候補領域を視覚的に表示）と、AIが生成したテキスト所見（例: 病変の有無、一行サマリ、詳細説明、信頼度、推定解剖学的部位など）を画面に表示します。ユーザーは画像上のハイライトと文章による説明を同時に得ることで、直感的かつ詳細に読影をサポートされます。
役割分担: このツールでは各コンポーネントが以下の役割を担います。
* Streamlit: ファイルアップロードや結果表示など、ユーザーインターフェース部分を担当します。シンプルなPythonコードでウェブUIを構築でき、迅速な試作に適しています。
* ANTsPyNet: MRI画像に対する専門的な前処理・解析を行います。例えばT1強調画像に対する脳抽出モデルを使い、頭蓋骨や不要領域を取り除いて脳領域を抽出したり、病変の疑いがある領域をセグメンテーションします。これによりノイズを減らし、後続のAI解析が注目すべき領域を絞り込めます。
* OpenAI GPT4.1 (Vision): アップロード画像とANTsPyNet解析結果（マスク画像）を入力とし、自然言語による所見記述や診断補助情報を生成します。GPT4.1は強力な画像認識と言語生成能力を備えており、構造化されたレポート出力も可能です。モデルには所定のプロンプトと言語指示を与え、JSON形式で機械可読な結果を返すようにします（例: 病変の有無、概要、詳細所見、信頼度、部位などの項目）。
* GitHub & Codex: 開発プロセスの管理とAI支援です。コードはGitHubでバージョン管理し、Pull Requestベースで進めます。コード補完やリファクタリングの際にGitHub CopilotやOpenAI CodexなどのAI支援ツールを活用します。本指示書は、そうしたAI開発アシスタントが誤解しにくいよう詳細かつ明確に作成されています。
以上の構成により、専門的な画像解析と高度なAI解釈を組み合わせた読影支援がローカル環境で実現できます。
2. 開発フェーズ
開発は段階を追って進め、段階ごとに機能追加と品質向上を図ります。以下の3フェーズに分けて計画します。
フェーズ1: ローカル環境でのプロトタイピング
目標: 最小機能の実装と動作検証を行います。まずはローカルPC上でStreamlitアプリを立ち上げ、一連のデータフロー（画像入力→ANTsPyNet処理→GPT出力→結果表示）が通ることを確認します。
* Streamlit UI構築: ユーザーが画像ファイルをアップロードするためのUIを用意します。st.file_uploaderウィジェットを使用し、MRI画像ファイル（例えばNIfTI形式の.niiファイルやPNG/JPEGなど単一スライス画像）を受け付けます。アップロード後、解析実行のためのボタン（「解析開始」など）を配置します。
* ANTsPyNetとの連携: アップロードされた画像ファイルをANTsPyNetで処理し、まずは脳抽出を試みます。ANTsPyNetが提供するプリトレイン済みモデルを使い、入力MRIから頭蓋外組織を除去した脳マスクを取得します。例えば、ANTsPyNetのbrain_extractionユーティリティ関数を呼び出し、得られた確率マップを閾値処理してバイナリマスク画像を生成します（詳細は「3.具体的なコーディング指示」を参照）。このフェーズでは、ANTsPyNet処理結果（マスク画像）をStreamlit上に表示し、ライブラリ連携が正しく行えることを確認します。
* OpenAI API出力の取得: OpenAIのPythonクライアントライブラリを用いてGPT4.1 APIに接続します。まずテキストプロンプトのみでGPT-4.1を呼び出し、固定のメッセージを表示できるか試します。その後、画像入力を含むマルチモーダルリクエストを試行します。GPT-4.1(Vision)が画像を解析しテキストを返す基本動作を確認することが目的です。最初は出力形式を厳密にせず自由応答でも構いません。例えば「このMRI画像から分かることを説明してください」というプロンプトを送り、応答テキストを取得・表示するところまで実装します。
* 出力の整形: GPTから得たテキストやデータを、Streamlit上で見やすく表示します。フェーズ1ではレイアウトやデザインは簡素で構いませんが、元画像とマスク、そしてGPTのテキスト出力が同時に見られる状態を作ります。これにより、解析パイプライン全体の動作をまず実証します。
チェックポイント: フェーズ1完了時点で、単一のMRI画像に対してツールが一連の解析を実行し、何らかの結果を画面表示できている必要があります。例えば、サンプルのMRI（正常例でも良い）を入力し、ANTsPyNetで脳部分が抽出された画像、およびGPT-4.1が生成した説明文が表示されれば成功です。
フェーズ2: エラーハンドリングとユニットテスト導入
目標: フェーズ1で構築したプロトタイプに対し、堅牢性の向上と品質保証を行います。具体的には、異常系への対応と自動テストの整備です。
* 入力ファイル形式の検証: ユーザーが誤った形式のファイルをアップロードした場合の対策を実装します。例えば、サポートしない拡張子（テキストファイルや破損した画像など）をアップした場合、ANTsPyNetの読み込み時にエラーが発生しうるため、処理開始前にファイルヘッダや拡張子を確認します。想定する入力は医療画像（DICOM, NIfTI, PNG/JPEG等）ですが、MVPではまず典型的なケース（例えば拡張子が.niiまたは.nii.gzのNIfTIファイル、およびサンプル用にPNG/JPEG）を許容し、それ以外はエラーとします。検証はuploaded_file.nameやPythonのimghdrモジュール等で行い、NGの場合はStreamlitの警告メッセージを表示して処理を中断します。
* ANTsPyNet処理エラーへの対処: 画像内容によってはANTsPyNetのモデルが収束しなかったり、予期せぬエラーを投げる可能性があります。そこで、ANTsPyNet呼び出し部分（例: brain_extraction関数）をtry/exceptで囲み、例外発生時にはNoneを返すかエラーメッセージをUI表示します。例えば「画像解析に失敗しました。画像形式や内容をご確認ください。」とユーザーにフィードバックします。エラー詳細はログに出力し、デバッグできるようにします。
* OpenAI API応答の妥当性チェック: GPT-4.1から期待するJSON構造で応答が常に得られるとは限りません。不完全なJSON文字列や予期しない回答が返る可能性があります。このため、API応答を受け取ったらまずjson.loadsでパースを試み、失敗した場合は例外を補足します。パースエラー時の扱いとして、再度プロンプトを送ってリトライする戦略や、エラー内容をUI表示する戦略が考えられます。MVPではまずエラーをユーザーに知らせ、必要ならログに生のAPI応答を記録する程度に留めます（将来的にはリトライやプロンプト調整による自動再試行も検討）。加えて、正常にJSONパースできた場合でも、期待するキーが存在するか（例えばis_finding_presentやconfidence_scoreが含まれているか）を検証し、欠落していればエラー処理を行います。Pydanticライブラリを用いて出力スキーマを定義しバリデーションする方法も有効です。Pydanticモデルを定義しておけば、LesionFinding.parse_obj(response_json)のように検証でき、不正な形式の場合は例外が出るため、それを捕捉して扱うことができます。
* ユニットテストの整備: 上記の様々なケースについて、自動テストを作成します。Pythonのunittestやpytestを用いて、主要な関数（画像解析関数、レポート生成関数など）を単体テストします。テスト項目の詳細は「6.テスト項目」で後述しますが、例えば「対応していないファイル形式を与えた場合に適切にエラーを出すか」「ANTsPyNet関数にダミー画像を与えて例外なく処理できるか」「GPT応答が不正な場合にNoneを返す処理が働くか」といったケースを網羅します。テスト用の軽量な画像データ（極小サイズのMRIまたはランダムノイズ画像など）を用意し、処理結果の型や値を検証します。
* コードリファクタリング: エラー処理ロジック追加に伴い、コードの見通しが悪くならないよう関数分割やリファクタを行います。例えば、ファイル検証・読み込み、画像解析、レポート生成、結果表示といった処理をそれぞれ独立した関数（もしくはモジュール）に分け、各関数は単一の責務を持つようにします。こうしておくと、テストもしやすくなり、将来的な改修も容易になります。
チェックポイント: フェーズ2完了時には、不正入力や処理失敗があってもアプリがクラッシュせず、ユーザーにわかりやすいエラー表示がなされる状態になります。また、自動テストを実行することで主要機能が期待通り動作することを確認できるようになります。
フェーズ3: 再現性の確保（構成管理、環境変数管理、要件定義）
目標: プロジェクトの再現性と保守性を高めるため、開発環境の標準化とプロジェクト構成の整備を行います。MVPをローカルのみならず他の環境でも動作させやすくし、将来的な機能追加にも耐えうる基盤を構築します。
* プロジェクトのディレクトリ構成: コードと資産を整理し、誰もが見て理解しやすいリポジトリ構造にします。以下は推奨する構成例です（詳細は「7.全体構造のテンプレート例」を参照）:
mri-reader-mvp/
├── app.py              # Streamlit アプリのエントリポイント
├── modules/            # 機能別のPythonモジュール群
│   ├── image_analyzer.py      # ANTsPyNetを用いた画像解析ロジック
│   ├── report_generator.py    # OpenAI APIを用いたレポート生成ロジック
│   └── __init__.py
├── tests/              # テストコード用ディレクトリ
│   ├── test_image_analyzer.py
│   ├── test_report_generator.py
│   └── ...
├── requirements.txt    # 必要なPythonライブラリ一覧（バージョン明記）
├── .env.example        # 環境変数設定のサンプル（OPENAI_API_KEYの項目など）
└── README.md           # プロジェクトの概要とセットアップ方法
コードは役割ごとにモジュールファイルへ分割します。例えば画像解析処理はimage_analyzer.pyに、レポート生成処理はreport_generator.pyに、それぞれ関数群として実装します。Streamlitのapp.pyでは主にUI構築と各モジュール関数の呼び出しに専念し、ロジックを持たせないようにします。この構成により、UIとビジネスロジックの分離が図れ、開発チーム内での並行作業やコードの見通しが向上します。
* 環境変数管理: APIキーなど機密情報や環境依存の設定値は、コード中に直書きせず環境変数から読み取るよう統一します。具体的には、OpenAIのAPIキーはOPENAI_API_KEYという名前で環境変数に保持し、Pythonコード内ではos.getenv("OPENAI_API_KEY")等で取得します。開発者がローカル環境で動かす際には、.envファイル（.env.exampleをコピーして作成）にキーを記載しておき、python-dotenvライブラリで読み込む運用も可能です。Gitリポジトリには.envファイルは含めず、あくまで.env.exampleに記載のみ行います。環境変数が未設定の場合の挙動も実装します（例: 起動時にキー未設定ならエラーを表示し終了する）。また、他の依存サービスのURLや設定値も必要に応じ環境変数化し、設定変更時にコード修正が不要なようにします。
* 依存パッケージ管理: requirements.txtにこのMVPで使用する全ての外部ライブラリを列挙し、バージョンを固定します。Python 3.11で動作確認しているため、TensorFlowやANTsPyNet、OpenAIライブラリ等のバージョンは互換性のあるものを選定します（例: antspynet==1.0.3、tensorflow==2.16.1 など）。こうすることで、開発者間や本番環境との環境差異による不具合を抑止します。将来的にはPoetryやpipenv等の導入も検討できますが、MVP段階ではシンプルにrequirements.txtで管理します。
* 再現性とキャッシュ: 本ツールはローカル実行が前提ですが、初回実行時にはANTsPyNetがモデルファイルをダウンロードする点に注意が必要です。一度ダウンロードされたモデルは通常ユーザーホーム配下（~/.antspynet/）にキャッシュされ、二回目以降の実行で再利用されます。よって、開発チーム内で共有する場合はこのキャッシュも含めておくか、初回起動時に自動的にダウンロードされる旨をREADMEに記載します。同様に、Streamlitの再実行時に毎回モデルロードやAPI呼び出しが走らないよう、適切にステートフルな管理をします。Streamlitには関数実行結果をキャッシュするst.cache_dataやst.cache_resource（旧st.cache）といった仕組みがあります。例えばANTsPyNetのモデル読み込みやOpenAI応答結果（必要なら）をキャッシュし、ユーザーが繰り返し同じ画像を解析する際の待ち時間を短縮できます。ただしメモリとのトレードオフもあるため、MVPでは必要最小限のキャッシュに留めます。また、ランダム性のある処理はシード固定（TensorFlowやNumPyの乱数シード設定）を行い、実行ごとに結果がぶれないようにします。これら工夫により、誰がどこで実行しても同じ結果が得られる再現性を確保します。
* ドキュメンテーション: README.mdにプロジェクトの概要、セットアップ手順、使用方法を記載します。特に、環境変数の設定方法（例: .envの用意と内容）や、ANTsPyNetモデルの初回ダウンロードについて明記します。また、開発者向けにはテストの実行方法（pytestコマンド等）や、コード構成の説明も記述します。コード内には要所でコメントを付し、複雑な処理や重要なロジックには日本語でコメントを残してAI補助ツールが誤解しにくいよう配慮します。
チェックポイント: フェーズ3完了時には、リポジトリをクローンしpip install -r requirements.txtして環境変数をセットすれば、どのマシンでも同じようにアプリを起動できる状態になります。また、環境設定やコード構成についての情報がREADMEやコメントで提供され、新しく参加した開発者やAIコード補助ツールでもスムーズに開発を進められるようになります。
3. 具体的なコーディング指示
ここでは、本MVPで実装すべき主要機能ごとに、具体的な実装方法や使用するライブラリ・関数について指示します。各項目には、Codex（GPTによるコーディング支援AI）でも誤解が生じにくいよう、明確なコメントと構造でコードを書くことを心がけます。
3.1 ファイルアップロード処理 (Streamlit UI)
* Streamlitによるファイル入力: st.file_uploaderを利用してユーザーから画像ファイルを受け取ります。例えば:
* uploaded_file = st.file_uploader("MRI画像をアップロードしてください", type=["nii", "nii.gz", "png", "jpg"])
* if uploaded_file is not None:
*     # 解析ボタンの表示
*     if st.button("解析開始"):
*         # 後続の処理呼び出し
*         analyze_and_report(uploaded_file)
ユーザーがファイルを選択するとuploaded_fileにストリームが格納されます。対応拡張子は必要に応じて調整します（NIfTIやDICOMも扱う場合は追加。ただしDICOMは複数ファイルにまたがるためMVPでは除外）。analyze_and_reportはアップロードファイルを受け取り解析〜表示を行う統合関数（または一連の処理）です。
* ファイルの保存またはメモリ展開: StreamlitのファイルアップロードはバイナリIOオブジェクトを返します。そのままANTsPyNetに渡すには、一度ファイルとして保存するかメモリ上で処理する必要があります。MVPでは簡便のため、一時ディレクトリ（例えばtempfileモジュールでtempfile.NamedTemporaryFileを使うか、Path("tmp/image.nii")など決め打ちパス）に保存し、そのパスをANTsPyNetで読み込む方法が確実です。もしくはANTsPyNetの読み込み関数がファイルオブジェクトを直接受け付けるならそれを利用します。いずれの場合も、ファイル名やパスにユーザー入力が反映されないよう注意します（セキュリティ上、安全な一時ファイル名を自動生成）。
* メタ情報抽出（必要に応じて）: 画像ファイルによっては解像度や次元数などのメタ情報が重要です。NIfTIなら3次元ボリュームですが、PNG/JPEGなら2D画像の想定です。ANTsPyNetのモデルはおそらく3DのT1 MRIを前提にしているため、2D画像を与えた場合はどのように扱われるか検証が必要です。MVP段階では、サンプルとして2Dの脳MRIスライス画像でも動作するよう調整します（例えば2Dを仮想的に3Dに拡張して入力するか、ANTsPyNetの2Dオプションがあれば使用）。そうした特殊ケースは一旦脇に置き、まずは正しい入力が来る前提で実装を進めます。後のテストで問題が出たら対処しましょう。
3.2 画像前処理とANTsPyNetによる脳抽出
* ANTsPyNet/ANTsライブラリのインポート: まずPythonコード先頭で必要なライブラリをインポートします。例えば:
* import ants
* from antspynet.utilities import brain_extraction
ANTsPyNetはANTs (ANTsPy)と連携して動作するため、antsモジュールも使用します。TensorFlowなどバックエンドフレームワークもインポートが必要です（ANTsPyNetは内部でTensorFlow/Kerasモデルを使用）。可能ならTensorFlowのログレベルを下げて（os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"など）冗長なログ出力を抑えます。
* 画像ファイルの読み込み: アップロードされたファイル（例えばNIfTI）をANTsイメージオブジェクトに変換します。ANTsPyのants.image_read関数はファイルパスまたはバイナリストリームから画像を読み込めます。一例として:
* import io
* def analyze_image(file_obj):
*     # ファイルバイトデータをメモリから読み込みANTsImageに変換
*     data = file_obj.read()
*     image = ants.image_read(io.BytesIO(data))  # メモリ上のデータから読み込み
*     # 脳抽出の実行
*     probability_mask = brain_extraction(image, modality="t1")
*     mask = ants.threshold_image(probability_mask, 0.5, 1.0)
*     return image, mask, probability_mask
上記では、ファイルのバイト列を取得してからio.BytesIO経由でANTsに渡しています（ANTsPyがBytesIOに対応していなければ、一時ファイル経由に変更します）。brain_extractionにはMRIのモダリティを指定（T1強調画像を想定）し、確率マップ（各ボクセルが脳領域である確率）を取得します。続いてants.threshold_imageで確率値0.5を閾値にマスク画像（0/1のバイナリ）を作成します。最終的に、元画像とマスクをタプルで返しています。必要に応じ、マスク画像から関心領域の座標（重心やバウンディングボックスなど）を計算し返すこともできますが、後段では画像データそのものをGPTに与えるため必須ではありません。
* エラーハンドリング: analyze_image内では、ANTsPyNet呼び出しを囲むtry/exceptを入れます。例えばbrain_extractionが内部で例外を送出した場合は、ログにスタックトレースを出力し（import tracebackしてtraceback.print_exc()等）、関数はNoneを返します。この関数を呼んだ側（Streamlitアプリ側）では戻り値をチェックし、Noneならユーザーに「画像解析に失敗しました」等のメッセージを表示します。なお、ANTsPyNetのモデル初回実行時はダウンロードが走るため時間がかかります。その旨をユーザーにインジケータ（st.spinner等）で知らせる、タイムアウトを設ける等の配慮も考えられます。
* 処理結果の確認: この脳抽出処理が正しく動作しているか、開発中に確認します。例えば、用意したテスト用MRI（可能なら脳MRIデータセットのサンプル）を入力し、mask結果をants.image_writeでファイルに保存してみたり、Streamlit上でst.image等を使って可視化します。ANTsPyNetの精度検証はMVP段階では深入りしませんが、明らかにおかしな結果（例えば真っ黒なマスクや全体が1のマスク）が出る場合は、モダリティ指定やANTsPyNetの別のモデル利用を検討します。ANTsPyNetのGitHubにはプリトレインモデル一覧があるので参考にします。
3.3 OpenAI APIへの画像＋プロンプト送信とJSON応答の取得
* OpenAI APIクライアントの設定: OpenAIのAPIキーを取得し、コード内で設定します。環境変数OPENAI_API_KEYからキーを読み込み、openai.api_keyにセットします。例えば:
* import openai
* import os
* openai.api_key = os.getenv("OPENAI_API_KEY")
* if openai.api_key is None:
*     st.error("OpenAI APIキーが設定されていません。環境変数OPENAI_API_KEYを設定してください。")
*     return
* 画像データの取り扱い: 現時点（2025年）で、OpenAIのGPT-4 APIは画像を直接引数として渡す正式なメソッドは公開されていませんが、Vision機能を利用する手段はいくつか考えられます。一般的なアプローチは、OpenAIのファイルアップロードAPIを使い画像ファイルを一時的にストレージに置き、そのファイルIDをChatCompletionで参照する方法です。具体的には、openai.File.create(file=open("image.png", "rb"), purpose="fine-tune")のようにして画像をアップロードし、返り値のidを取得します。そのfile_idを含む形でGPT-4モデルにメッセージを送ります（例えば「以下の画像 (ファイルID: X) を解析してください...」というプロンプト内で埋め込みます）。この方法により、GPT-4に画像内容を考慮させることができます。将来的にOpenAIのAPIが直接マルチモーダル入力をサポートすれば、その方法（例えばopenai.ChatCompletion.create(..., files=[...])のようなインターフェース）が利用できますが、MVP開発時点では上記のファイルストレージ経由の手段を取ります。
* プロンプト設計: 画像（MRIとマスク）の内容を適切にGPTに解釈させ、望むJSON構造で回答させるため、システムメッセージとユーザーメッセージを組み合わせて丁寧に指示します。システムメッセージではAIに与える役割や口調を定義します（例: 「あなたは医療画像解析を支援する有能なAIです。慎重かつ正確に質問に答えてください。」）。ユーザーメッセージではタスクの具体を指示します。例えば:
* システム: あなたは医療画像解析を支援するAIアシスタントです。
* ユーザー: 2枚の画像を提供します。1枚目は患者の脳MRI画像（T1強調）、2枚目はANTsPyNetにより算出された病変確率マップです。
* これらを分析し、以下の項目についてJSON形式で回答してください：
* - is_finding_present: 病変が存在する場合はtrue、無ければfalse
* - finding_summary: 病変がある場合、その所見の簡潔なサマリー
* - detailed_description: 病変がある場合、位置・大きさ・信号強度など詳細
* - confidence_score: 所見に対する信頼度を0.0〜1.0で
* - anatomical_location: 病変がある場合、おおよその解剖学的部位
* 病変が全く無い場合は、is_finding_presentをfalseとし他のフィールドはnullで回答してください。
このようなプロンプトに続けて、実際の画像ファイルIDや画像自体を添付します（技術的には前述のファイルアップロードIDを文中に含めるか、API機能で画像バイナリそのものを送信）。モデルに対して出力形式を厳密に指示している点が重要です。例えば「上記のフォーマットに従い、余計な説明を一切加えずにJSONのみを出力してください」と付記し、モデルが純粋なJSONだけを返すよう促します。Codex等でコードを書く際には、このプロンプト文字列をコード上で適切にエスケープ（改行やクオートに注意）する必要があります。
* APIコールとレスポンス取得: Pythonではopenai.ChatCompletion.createを使用してGPT-4モデルにメッセージを送り、レスポンスを受け取ります。例:
* response = openai.ChatCompletion.create(
*     model="gpt-4-vision",  # 仮にGPT-4 Vision対応のモデル名
*     messages=messages,     # 上記で構築したメッセージのリスト
*     temperature=0.0        # 精度重視のため低温度設定
* )
* result_text = response["choices"][0]["message"]["content"]
ここでは温度を0に設定し、出力のブレを減らしています。result_textにはモデルの出力したJSON文字列が入る想定です。
* JSONパースとバリデーション: 受け取った文字列をjson.loads(result_text)でパースし、Pythonのdictに変換します。前述のとおり、この際に例外処理を行います。具体的には:
* import json
* try:
*     report_data = json.loads(result_text)
* except json.JSONDecodeError as e:
*     print("JSONデコード失敗:", e)
*     report_data = None
report_dataがNoneの場合は、UIに「AIからの応答を解析できませんでした」と表示します。加えて、PydanticモデルLesionFindingを定義している場合は、例えばLesionFinding.model_validate(report_data)（Pydantic v2の場合）を呼び出し、フィールド型の妥当性チェックを行います。この時点で求めるデータ構造（boolやfloatの型など）が保証されるため、以降の処理が安全になります。
* エラー時のフォールバック: OpenAI API呼び出しがエラーを返す（例: APIキー不正・モデル利用不可・レート制限超過）場合の対処も実装します。openai.ChatCompletion.createは例外（例えばopenai.error.OpenAIError系）を投げる可能性があるため、これを捕捉しUIにエラー内容を表示します。Vision機能が利用できない場合（例えばAPIキーに画像権限がない、または一時的に失敗）は、フォールバック処理としてテキストのみの応答を試みることが考えられます。具体的には、画像なしで「テキストで所見を説明してください」というプロンプトを送り、可能な範囲でテキスト所見を得る、あるいはANTsPyNetが出力したマスクの統計情報（「明るい領域はどこそこにあります」等）を自前で算出し、それをテキスト化してユーザーに提供する方法があります。しかしMVP段階では無理に代替出力を生成せず、「画像解析AI機能が利用できません（OpenAI Vision機能にアクセスできません）」等と通知するだけでも良いでしょう。重要なのは、何も起きないままフリーズしたりクラッシュしたりしないことです。適切なメッセージを出してユーザーに再試行や設定見直しを促す実装とします。
3.4 Streamlit上での結果表示
* 画像とマスクの視覚的表示: Streamlitではst.imageを使って画像を表示できます。ただしMRIのような医用画像（グレースケール、かつ範囲がHU値やMRI信号値だったりする）はそのままだと可視化が難しい場合があります。簡易的には、ANTsPyNetから得られたimageおよびmaskをants.image_writeで一旦PNGなどに書き出してから読み込み表示するか、image.numpy()でnumpy配列にしてst.imageに渡します。例えば:
* # antsimageをnumpyに変換（0-255にスケーリング）
* import numpy as np
* brain_np = image.numpy()
* brain_np = (brain_np - brain_np.min()) / (brain_np.max() - brain_np.min()) * 255
* brain_np = brain_np.astype(np.uint8)
* mask_np = mask.numpy().astype(np.uint8) * 255  # 0 or 255
* # カラー合成（マスク部分を赤くするなど）
* import cv2
* colored = cv2.merge([brain_np, brain_np, brain_np])
* colored[mask_np > 0] = [255, 0, 0]  # マスク領域を赤色に
* st.image(colored, caption="脳抽出+病変マスク重畳表示")
上記はOpenCVを用いて、グレースケール画像を擬似的にカラー3チャンネルにし、マスク部分を赤く塗った例です。シンプルには、元画像とマスク画像を別々にst.imageで並べて表示する方法もあります。MVPではまず正しく領域を視覚化できることを重視し、デザインは凝りすぎないようにします。必要ならStreamlitのst.pyplotでmatplotlibを使い、透過マスクを重ねる等も可能です。
* テキスト所見とメタ情報の表示: GPT-4.1から得られたJSON構造の所見データは、項目ごとに整形して表示します。例えば:
* if report_data:
*     if report_data["is_finding_present"]:
*         st.markdown(f"**所見要約:** {report_data['finding_summary']}")
*         st.markdown(f"**詳細説明:** {report_data['detailed_description']}")
*         st.markdown(f"**推定部位:** {report_data['anatomical_location']}")
*         st.markdown(f"**信頼度:** {report_data['confidence_score']:.2f}")
*     else:
*         st.markdown("**所見:** 特に異常は検出されませんでした。")
Streamlitではst.writeやst.markdownでテキストを表示できます。Markdownを使うことで太字や改行を整形しています。数値はフォーマット指定で小数点以下2桁など見やすくします。日本語と数値・英単語が混ざる場合もフォーマットに気を配ります（例えばf文字列で括ることで文字化けを防ぐ）。また、必要に応じて追加情報（処理時間やモデルのバージョン等）を表示しても良いでしょう。
* ユーザー操作性の向上: Streamlitのインタラクティブ性を活かし、例えば解析結果を保存するボタン（JSONをダウンロードできるようにする等）を付けたり、別画像をすぐ解析できるようサイドバーにファイルアップローダーを置くなどUI工夫も可能です。ただしMVPでは基本的な一本のフローだけをまず実装します。レイアウト調整はst.columnsを使って画像とテキストを左右に並べる、などシンプルな改善を行います。
また、読影支援ツールとしての注意書き（例: 「本結果はAIによる補助情報であり、診断は専門医が行ってください」といったDisclaimer）も画面下部に表示しておくと良いでしょう。これは医療AIとしての倫理面配慮ですが、開発指示書にも盛り込んでおきます。
4. APIキーの読み取りと安全管理
本ツールで使用するOpenAI APIキーは、セキュリティのため環境変数で管理します。開発時および運用時において、APIキーがハードコーディングされないよう十分注意してください。
* 環境変数からの読み取り: Pythonではos.getenv("OPENAI_API_KEY")やos.environ["OPENAI_API_KEY"]で環境変数を取得できます。前者は変数未設定時にNoneを返すため安全です（後者は未設定だとKeyError）。openai.api_keyに直接代入することで以降のAPI呼び出しにキーが適用されます。なお、環境変数の設定は開発者各自のPCで行う必要があります。Windowsならシステム環境変数に登録、macOS/Linuxなら~/.bashrc等にexport OPENAI_API_KEY="sk-..."を追記します。プロジェクトのREADMEや.env.exampleに、この変数設定方法を記載してください。
* 未設定時の対処: APIキーが設定されていない場合、アプリ起動時にただちにエラーが分かるようにします。例えばopenai.api_key = os.getenv("OPENAI_API_KEY")の後に、値がNoneならst.error("APIキーが見つかりません。環境変数OPENAI_API_KEYを設定してください。")を表示し、returnやsys.exit(1)で処理を止めます。Codexでコードを書く際も、このチェックを入れるコメントを忘れないようにしましょう。開発中にキーを誤ってコミットしそうになった場合、Gitの履歴から確実に削除すること（GitHub上に一度でもキーが上がると無効化する必要があります）。
* Git管理から除外: APIキーは決してGitリポジトリに含めません。.envファイルを使用する場合は、リポジトリのルートに.gitignoreを作成し、その中に.envを記述して追跡対象外にします。で述べられているように、環境変数を使うことでキーをコードや設定ファイルに残さずに済み、誤って公開リポジトリに載せてしまう事故を防げます。
* 権限とローテーション: OpenAI APIキーはアカウントごとに発行されますが、万一漏洩した場合に備え、必要に応じて定期的にキーを再発行（ローテーション）してください。また、プロジェクトメンバー間でキーを共有する場合は、Slackなどで平文で送らず、1Passwordのような秘密情報共有ツールを使うことが望ましいです（MVP開発ではメンバー各自で鍵を設定するだけで足ります）。
* API使用量の監視: キー管理とは少し異なりますが、安全な使用のためにレート制限や費用にも注意します。OpenAI APIには利用上限があるため、必要以上に高頻度で呼び出さない工夫（例えばユーザー操作に対して連打対策のUI制御をする、結果をキャッシュするなど）も検討します。また、OpenAIのダッシュボードでAPI使用量を確認し、異常な利用がないかモニタリングします。このように、安全かつコスト効率よくAPIキーを運用してください。
5. Codex向けにエラーを引き起こしにくい工夫
本プロジェクトではAI補助によるコーディングを前提としています。GitHub CopilotやOpenAI Codex等を効果的に使うために、AIが誤解しにくいコード構造と記述を心がけます。以下に、具体的な工夫ポイントをまとめます。
* 文字コードと文字列リテラル: ソースファイルはUTF-8で保存し、ファイル先頭に# -*- coding: utf-8 -*-を明記することで、非ASCII文字（日本語コメントや文字列）が含まれてもエンコーディング問題が発生しないようにします。プロンプト内に日本語を含める必要があるため、Unicodeでのエスケープに注意します。Codexはしばしばバックスラッシュやクォートを誤ってエスケープする可能性があるので、f文字列や三連引用符を適切に使い、読みやすい文字列連結にします。例えば、長いプロンプトは(""" ... """)のように括り、変数を埋め込みたい場合は適宜フォーマット関数を使います。AIが生成したコードでエスケープミスが起きたら、人間がすぐに検知できるよう、日本語部分を含めて単体テスト（例えばプロンプト生成関数がエラーなく文字列を返すか）も書いておきます。
* インポート順序と依存関係: Pythonではインポート順序によっては依存関係の初期化が必要なケースがあります。例えばANTsPyNetはTensorFlowより先にインポートしたほうが良い（GPUメモリ確保の順序など）場合があります。確実を期すため、標準ライブラリ→サードパーティ（antspynet等）→自作モジュールの順でimportを書くようにします。Codexもその順序を学習している可能性があるため、コメントで区切って順序を書くと良いでしょう。例:
* # 標準ライブラリ
* import os, io, json
* # サードパーティライブラリ
* import ants
* from antspynet.utilities import brain_extraction
* import openai
* import streamlit as st
* # 自作モジュール
* from modules import image_analyzer, report_generator
こうしたひな形を最初に示しておけば、AIもそれに倣ってコードを補完しやすくなります。
* グローバル変数の扱い: Streamlitアプリではスクリプトが再実行されるたびにグローバル変数が初期化される点に留意します。極力グローバルな可変状態は持たない実装にします。例えば、APIから取得した結果を一時保存するグローバル変数を使うより、必要ならst.session_stateを活用します。Codexにコードを書かせる際も、「グローバル変数は使わない」「副作用を持たない関数を作る」というコメントを前もって与えると良いでしょう。また、ANTsPyNetのモデルやOpenAIの設定などは、一度読み込んだら再利用できるように、st.cache_resourceを使ってキャッシュし、再実行時にも状態を引き継ぐようにします。AIがこれを自動で考慮しにくい部分なので、人間側で設計を固定し、適宜コメントで指示します。
* ステートフローの明示: Codexは文脈からコードを推測しますが、Streamlit特有の再実行やインタラクションの流れは誤解のもとになります。そこで、「まずこの関数を呼び、その中でこれらの処理を行う」という制御フローをコメントやDocstringで明示します。たとえばanalyze_and_report()関数の冒頭に「# 1) 画像解析 2) レポート生成 3) 結果表示 を順に実行するメイン関数」とコメントしたり、Docstringに使用例を書くと、Codexがより正確に意図を汲んでくれます。
* 繰り返し実行時のキャッシュとリソース管理: Streamlitアプリはユーザー操作のたびにスクリプト全体を実行します。そのため、ANTsPyNetのモデルロードやOpenAI API設定は、毎回繰り返さないように工夫します。上述のst.cacheのほか、モジュールロード時に一度だけ処理するコード（例えばグローバルにantspynet.utilities.get_pretrained_networkを呼んでモデルを事前ロードする等）を書いておく方法もあります。しかし、長時間メモリを占有する懸念もあるため、MVPでは一旦シンプルに都度ダウンロードに任せ、後で必要なら最適化します。Codexには「# TODO: モデルの初回ダウンロードをキャッシュする」等コメントを残し、将来の改善点を示します。これにより、AIが中途半端に最適化しようとして複雑なコードを生成するのを防ぎます。
* エラー内容の明示: コード内で例外処理する際、Exceptionを無視せずログ出力やst.errorで内容を表示するようにします。Codexは時にexceptブロックを空にしたりpassしてしまうことがあるため、明示的に「# エラー内容を出力する」など指示します。例えば:
* try:
*     result = brain_extraction(image, modality="t1")
* except Exception as e:
*     st.error(f"画像解析中にエラー: {e}")
*     return None
とし、どんな例外でも捕捉してユーザーに知らせます。このように実装すれば、Codexの提案したコードでも重要なエラー情報が失われません。
6. テスト項目
本MVPに対して想定されるユースケースおよびエッジケースについて、以下の観点でテストを行います。テストは自動化（pytest等）できるものは自動化し、難しいものは手動検証します。可能な限り単体テストと統合テストを用意し、リグレッションを防ぎます。
* 画像未入力時の挙動: ユーザーが画像をアップロードせずに解析ボタンを押した場合（またはアップロード後にキャンセルした場合）を想定します。このとき、analyze_and_reportが呼ばれない or 早期リターンすることを確認します。UI上は「ファイルが選択されていません」という警告を表示するなどの対応が望ましいです。自動テストでは、Streamlit部分はモックしつつanalyze_image(None)が呼ばれないことを検証します。
* 対応外ファイル形式の入力: BMP画像やテキストファイルなど、サポートしない形式をアップロードした場合、適切にエラーメッセージが表示されるかテストします。例えば.txtファイルを仮想的にuploaded_fileに見立てて渡し、関数内で拡張子チェックが機能するかを見ます。期待結果は関数が例外を投げるか、FalseやNoneを返し、UIに「未対応のファイル形式です」と表示されることです。
* 異常または空の画像入力: サイズが0のファイル、全画素値が同じ画像、極端に大きな画像などを与えたときの挙動を確認します。ANTsPyNetのbrain_extractionがこれらで落ちないか、落ちた場合のハンドリングが正しく動くか見ます。例えば全ゼロ画像を与えたときにis_finding_presentがfalseになるか、極端に大きな画像でメモリエラーが起きないか（これは厳密には性能テスト領域ですが）。MVPでは性能より機能テストを重視します。
* ANTsPyNet処理結果の検証: 正常な脳MRI画像を入力した場合に、脳領域マスクが生成されていることをテストします。自動テストでは難しいので、小さな画像でanalyze_imageを呼び出し、戻り値がANTsImageオブジェクト2つになっているか、マスクの画素値が0/1であるかなどを確認します。ユニットテストではANTsPyNetの機能自体を信頼しつつ、関数のインターフェースを検証する形になります。
* OpenAI API応答エラーハンドリング: APIキー不正・ネットワーク不通・応答遅延などを模擬し、例外処理が正しく動くかテストします。これはOpenAI APIを実際にコールせず、モンキーパッチやモックでopenai.ChatCompletion.createが例外を投げるように仕込みます。そしてgenerate_report関数（仮称）を呼び、戻り値がNoneになるか、エラーメッセージが所定の内容で表示されるか確認します。
* GPT応答JSON内容の検証: GPTから返ってきたJSON文字列をパース・バリデーションする部分のテストを行います。想定される正常JSONや、括弧が欠けている不完全JSON、想定外のキーを含むJSONなどいくつか用意し、それらに対してパース処理がどう結果を返すか検証します。例えば、'{"is_finding_present": true, ...}'という文字列なら正しくdictになるはず、'{"is_finding_present": "yes"}'のように型が違う場合Pydanticがエラーを出すはず、などをチェックします。
* Vision未対応時のフォールバック: GPT-4のVision機能が使えないケースをテストします。例えばモックで「このモデルは画像入力に対応していません」というメッセージを返すようにし、それを受けて我々のコードがNoneや適切なエラーを返すか確認します。また、フォールバックとしてテキストのみの出力を試みる実装をした場合、その結果が一貫した形式（例えばis_finding_present=falseのみのJSONなど）になるか、人間が目視で確認します。MVPの段階ではフォールバックは限定的なため、自動テストより手動確認になるかもしれません。
* 全体統合テスト（手動）: ローカルPC上で実際にStreamlitアプリを起動し、ブラウザから正常ケース・異常ケースを一通り操作して確認します。特に、典型的な病変があるMRI画像（もし用意できれば）を投入し、AIの返す所見がそれらしく妥当か、人間の知見と照らし合わせます。ここで精度の議論までは不要ですが、「明らかに見当外れな出力ではないか」「JSONが表示崩れしていないか」「複数回実行してもクラッシュしないか」等をチェックします。
以上のテスト項目を順次実施し、不具合が見つかればIssueを起票して修正します。特にエラーハンドリング周りは一見動いているようでも網羅漏れがちなので、時間をかけて潰します。ユニットテストと手動テストを組み合わせ、堅牢なMVPに仕上げましょう。
7. 全体構造のテンプレート例とコード断片
最後に、プロジェクト全体のコード構造のテンプレートと、各主要モジュールのコード例（抜粋）を示します。開発者やAI補助ツールが参照できるひな型として活用してください。コードには理解を助けるコメントを添えています。
ディレクトリ構成テンプレート
前述した構成を改めて示します。
mri-reader-mvp/
├── app.py
├── modules/
│   ├── __init__.py
│   ├── image_analyzer.py        # 画像前処理・解析（ANTsPyNet利用）
│   ├── report_generator.py      # レポート生成（OpenAI API利用）
│   └── visualization.py         # 結果可視化処理（必要に応じて分離）
├── tests/
│   ├── test_image_analyzer.py
│   ├── test_report_generator.py
│   └── test_end_to_end.py       # 統合テスト用（Streamlit抜きで関数を直呼びするなど）
├── requirements.txt
├── .env.example
└── README.md
* app.py: Streamlitアプリのエントリポイントです。UIレイアウトの定義と、ユーザー操作に応じたモジュール関数呼び出しを記述します。複雑な処理ロジックは記載せず、ほぼ「グルーコード」として働きます。
* modules/image_analyzer.py: ANTsPyNetを使った画像処理関連の関数をまとめます。analyze_image(file_obj)など前処理からマスク取得までを実装します。
* modules/report_generator.py: OpenAI APIを使ったレポート（所見JSON）生成の関数をまとめます。generate_report(image, mask)関数では、環境変数からAPIキー取得、プロンプト構築、API呼び出し、JSONパースまで行います。
* modules/visualization.py: 必要なら画像とマスクの合成や着色処理、Streamlit表示用ユーティリティをここに入れます。シンプルな処理であればapp.pyに書いても構いません。
* tests/: 単体テストと必要に応じ統合テストのスクリプト群です。pytestで実行すれば各関数の期待動作を検証できます。将来的にはStreamlitをヘッドレスブラウザで起動しての集成テストも可能ですが、MVPでは関数レベルのテストに留めます。
コード断片例
以下に、本プロジェクトの主要機能のコード断片を例示します。実際の実装の際には、この例を土台にしつつ、エラー処理やログ出力を適宜補完してください。
1. 画像解析 (image_analyzer.py)
import io
import ants
from antspynet.utilities import brain_extraction

def analyze_image(file_obj):
    """
    アップロードされた医用画像ファイルを解析し、
    脳抽出マスクを生成する関数。
    """
    # ファイルをANTsImageとして読み込む
    try:
        data = file_obj.read()
        image = ants.image_read(io.BytesIO(data), pixeltype='float')
    except Exception as e:
        # 画像読み込み失敗
        print(f"[Error] 画像読込エラー: {e}")
        return None

    try:
        # 脳抽出を実行（T1w MRIを想定）
        probability_mask = brain_extraction(image, modality="t1")
        # 確率マップを閾値0.5で二値マスクに
        mask = ants.threshold_image(probability_mask, 0.5, 1.0)
    except Exception as e:
        # 解析中にエラー（モデル未ダウンロード等）
        print(f"[Error] ANTsPyNet処理エラー: {e}")
        return None

    # 結果を返却（ANTsImageオブジェクト3つ）
    return image, mask, probability_mask
上記コードでは、ファイルオブジェクトから直接画像を読み込んでいます。ANTsPyNetのbrain_extractionモデルが内部でTensorFlowを使用するため初回時間がかかる点に留意してください。また、必要であればantspynet.utilities.get_pretrained_network('brainExtraction')をどこかで一度呼び出しておくと、事前にモデルダウンロードが可能です。
2. レポート生成 (report_generator.py)
import os, json
import openai

# Pydanticのスキーマ定義（省略も可）
# from pydantic import BaseModel
# class LesionFinding(BaseModel):
#     ...

def generate_report(image, mask):
    """
    脳MRI画像と病変マスクをもとにGPT-4.1に所見JSONを生成させる。
    """
    # APIキー設定
    openai.api_key = os.getenv("OPENAI_API_KEY")
    if openai.api_key is None:
        raise RuntimeError("OpenAI APIキーが設定されていません。")

    # 画像をOpenAIに送る準備（ファイルアップロード）
    try:
        ants.image_write(image, "/tmp/original.png")
        ants.image_write(mask, "/tmp/mask.png")
    except Exception as e:
        print(f"[Error] 画像保存エラー: {e}")
        return None
    try:
        file1 = open("/tmp/original.png", "rb")
        file2 = open("/tmp/mask.png", "rb")
        res1 = openai.File.create(file=file1, purpose="assistants")
        res2 = openai.File.create(file=file2, purpose="assistants")
        file1.close(); file2.close()
        file_id1 = res1["id"]; file_id2 = res2["id"]
    except Exception as e:
        print(f"[Error] OpenAIファイルアップロード失敗: {e}")
        return None

    # プロンプト構築
    system_prompt = "あなたは医療画像解析を支援する有能なAIアシスタントです。"
    user_prompt = (
        f"1枚目の画像(ID: {file_id1})は患者の脳MRI画像、"
        f"2枚目の画像(ID: {file_id2})は病変確率マップです。"
        "これらを解析し、JSON形式で以下の項目について回答してください:\n"
        "- is_finding_present: ... (true/false)\n"
        "- finding_summary: ...\n"
        "- detailed_description: ...\n"
        "- confidence_score: ... (0.0-1.0)\n"
        "- anatomical_location: ...\n"
        "出力はJSONのみを返し、説明は不要です。"
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    # API呼び出し
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4-vision",  # GPT-4.1 Visionモデル（仮称）
            messages=messages,
            temperature=0.0
        )
    except Exception as e:
        print(f"[Error] OpenAI API呼び出し失敗: {e}")
        return None

    result_text = response["choices"][0]["message"]["content"]
    # JSONパース
    try:
        result = json.loads(result_text)
        # （必要ならここでPydanticで検証）
    except Exception as e:
        print(f"[Error] OpenAI応答JSON解析エラー: {e}")
        return None

    return result
このコード断片では、まずANTsImageをPNGに保存し、それをOpenAIにアップロードしています（Stack Overflowの報告例を参考にpurpose="assistants"でアップロードしています）。取得したfile_idをプロンプト内に埋め込んでGPT-4に渡しています。モデル名"gpt-4-vision"は仮のものです。実際にはOpenAIが指定するモデルエンジン名を使用してください（ドキュメントで確認）。APIレスポンスからcontentを取り出し、jsonにパースするところまで実装しています。各段階で例外をキャッチし、ログ出力している点に注目してください。
3. Streamlitアプリ統合部 (app.py)
import streamlit as st
from modules import image_analyzer, report_generator

st.title("MRI読影支援ツール (GPT-4.1 + ANTsPyNet)")
uploaded_file = st.file_uploader("MRI画像ファイルを選択してください", type=["nii", "png", "jpg"])
if uploaded_file:
    if st.button("解析開始"):
        # 画像解析
        with st.spinner("画像解析中...(ANTsPyNet)"):
            result = image_analyzer.analyze_image(uploaded_file)
        if result is None:
            st.error("画像の解析に失敗しました。対応していない形式か、画像内容に問題がある可能性があります。")
            st.stop()
        image, mask, prob = result

        # レポート生成
        with st.spinner("AIレポート生成中...(GPT-4.1)"):
            report = None
            try:
                report = report_generator.generate_report(image, prob)
            except Exception as e:
                st.error(f"AIレポート生成中にエラー: {e}")
        if report is None:
            st.error("AIが画像を解析できませんでした。")
            st.stop()

        # 可視化と出力表示
        st.subheader("解析結果")
        # 画像とマスクの表示（簡易版）
        st.image(ants.plot(image, mask, overlay_alpha=0.5), caption="原画像とマスク")  # ants.plotは画像をPILで返す関数
        # テキスト結果表示
        if report.get("is_finding_present"):
            st.markdown(f"**所見要約:** {report['finding_summary']}")
            st.markdown(f"**詳細:** {report['detailed_description']}")
            st.markdown(f"**推定部位:** {report['anatomical_location']}")
            st.markdown(f"**信頼度:** {report['confidence_score']:.2f}")
        else:
            st.markdown("**所見:** 異常所見は検出されませんでした。")
上記はStreamlitアプリの主要部分です。ファイルアップロード→解析ボタン押下→スピナー表示しつつ解析→結果表示という流れを示しています。ants.plotという関数（ANTsPyが提供する可視化用関数）がある場合は利用できますし、なければ前述の方法で画像合成して表示します。st.stop()は以降の処理を停止するStreamlit固有の関数で、エラー時に後続の表示処理をスキップするために使っています。
コード断片の各所にコメントを書き、処理意図を明示しています。実際の実装ではこのテンプレートをベースに細部を調整してください。

以上が、GPT-4.1とANTsPyNetを活用したMRI読影支援MVPの開発指示書です。段階的な開発計画に沿って実装とテストを進めれば、ローカル環境で動作する有用なプロトタイプが完成するでしょう。開発中はCodex等のAIアシスタントを積極的に活用しつつも、本書のガイドラインを遵守し、人間のレビューとテストを経て品質を担保してください。健闘を祈ります！

