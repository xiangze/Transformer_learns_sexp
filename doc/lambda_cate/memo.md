# D代数とKl(D)について詳しく説明してください
分布モナド $D$、その Kleisli 圏 $\mathrm{Kl}(D)$、そして $D$-代数(Eilenberg–Moore 圏)を、定義から Transformer への接続まで順に説明します。これまで「softmax = $\mathrm{Kl}(D)$ の射」「value 混合 = $D$-代数の作用(期待値)」と繰り返してきたものの中身です。

## 1. 分布モナド $D$
台圏を $\mathbf{Set}$ とし、有限台版で定義します(連続版は後述)。
$$D(X)=\Big\{\,p:X\to[0,1]\ \Big|\ \mathrm{supp}(p)\text{ 有限},\ \textstyle\sum_{x}p(x)=1\,\Big\}$$

$X$ 上の有限台確率分布の集合です。これがモナドになる三点セット:

- **単位** $\eta_X:X\to D(X)$、$x\mapsto \delta_x$(ディラック測度、点質量)。「確定値を分布とみなす」。
- **乗法** $\mu_X:D(D(X))\to D(X)$、分布の分布を平均で潰す:$\mu(P)(x)=\sum_{q}P(q)\,q(x)$。**全確率の法則**そのもの。
- **関手作用** $D(f):D(X)\to D(Y)$($f:X\to Y$)は**押し出し**:$D(f)(p)(y)=\sum_{x:f(x)=y}p(x)$。

モナド則(左右単位律・結合律)は、ディラックの周辺化と混合の結合性という**確率の基本恒等式**に一致します。$\eta$ が「確定化」、$\mu$ が「混合の平坦化」。

台圏$D$ は**可換**(独立な二分布のサンプル順序が結果を変えない=Fubini)かつ**アフィン**($D(1)\cong 1$、一点集合上の分布は一つだけ)。この二つが後で $\mathrm{Kl}(D)$ を Markov 圏にします。

**変種**:可測空間 $\mathbf{Meas}$ 上の **Giry モナド** $G(X)=\{X$ 上の確率測度$\}$(連続版、attention の実数値ロジットにはこちら)、部分確率の subdistribution モナド($\sum\le 1$)など。softmaxは有限台なので$D$で十分です。

## 2. Kleisli 圏 $\mathrm{Kl}(D)$ ― 確率核の圏

$\mathrm{Kl}(D)$ は「確率的射」を主役にする圏です。
- **対象**:台圏と同じ(集合)。
- **射** $X\to Y$:関数 $X\to D(Y)$、すなわち各 $x$ に $Y$ 上の分布を割り当てる **Markov 核(確率核)**。有限集合なら列和(または行和)1 の**確率行列** $k(x)(y)=P(y\mid x)\ge0$。
- **恒等射**:$\eta_X$、$x\mapsto\delta_x$(確定的にそのまま)。
- **合成**(Kleisli 合成 = **Chapman–Kolmogorov**):
$$(k'\circ k)(x)(z)=\sum_{y}k(x)(y)\,k'(y)(z)$$
確率行列の積。「$x$ から $y$ へ、$y$ から $z$ への遷移を中間 $y$ で周辺化」。
- **対称モノイダル構造**($D$ の可換性から):$\otimes=$ 集合の直積、核のテンソル = 独立同時分布。
- **Markov 圏構造**:copy $X\to X\otimes X$、$x\mapsto\delta_{(x,x)}$(確定的複製)と delete $X\to 1$。$D$ がアフィンゆえ delete が一意(semicartesian)、可換ゆえ対称モノイダル。**この copy は非自然** ― 真に確率的な核 $k$ では「copy してから $k$」($(y,y)$ の完全相関ペア)と「$k$ してから copy」(独立再サンプル)が一致しない。相関を生む copy です。

有限集合に制限した $\mathrm{Kl}(D)$ が **FinStoch**(確率行列の圏)。**softmax$(QK^\top)$ はまさにこの圏の射** ― クエリ位置 → 鍵位置の確率核 $A$、確率行列です。

## 3. $D$-代数(Eilenberg–Moore 圏 $\mathrm{EM}(D)$)

Kleisli が「射」を主役にしたのに対し、$D$-代数は「**分布を消費して値を返す構造**」を主役にします。

**$D$-代数**とは組 $(X,\alpha)$、$\alpha:D(X)\to X$ で、二つの整合則:

$$\alpha\circ\eta_X=\mathrm{id}_X\qquad(\text{点質量 }\delta_x\text{ は }x\text{ に評価される})$$
$$\alpha\circ\mu_X=\alpha\circ D(\alpha)\qquad(\text{分布の分布は、平坦化してから評価 = 各々評価してから評価})$$

**意味**:$\alpha$ は「$X$ 上の分布を一つの元へ」写す、すなわち**重心・期待値・凸結合を取る**操作。つまり $D$-代数 = 凸結合が取れる集合。

**具体的に何が $D$-代数か**:有限分布モナドの Eilenberg–Moore 圏は**凸空間(abstract convex spaces)**の圏と同値です(Fritz–Perrone ら)。私たちにとって決定的な例 ― **任意の実ベクトル空間 $V$ は $D$-代数**で、その構造射は

$$\alpha_V:D(V)\to V,\qquad \alpha_V(p)=\sum_{v}p(v)\,v=\mathbb E_{p}[v]\quad(\textbf{期待値})$$

より一般にベクトル空間の凸部分集合も $D$-代数。**代数の射は affine 写像**(凸結合と可換 = $\alpha$ と可換)。

**value 混合はこの作用そのもの**:softmax が $A(x)\in D(\text{位置})$ を出したあと、value 混合 $A\!\cdot\!V=\sum_y A(x)(y)\,v_y$ は、値空間 $V$ の $D$-代数構造 $\alpha_V:D(V)\to V$ を**分布に適用したもの = 期待値**。だから

$$\text{attention head}=\underbrace{\big[\mathrm{Ctx}\xrightarrow{\ \text{softmax}\ }D(\text{pos})\big]}_{\mathrm{Kl}(D)\text{ の射}}\ \text{に}\ \underbrace{\big[D(V)\xrightarrow{\ \alpha_V=\mathbb E\ }V\big]}_{D\text{-代数の作用}}\ \text{を合成}.$$

Kolmogorov との整合:確率変数 = 標本空間 $\Omega\to\mathbb R$ の可測関数、期待値 $\mathbb E$ が $\mathbb R$(や $V$)の $D$-代数(Giry-代数)構造。**softmax が route し、期待値が mix する**、が「核 ∘ 代数作用」という一文に収まります。

## 4. $\mathrm{Kl}(D)$ と $\mathrm{EM}(D)$ の関係 ― 随伴と比較関手

両者は無関係でなく、**同じモナド $D$ の二つの解決(resolution)**です。任意のモナドは随伴 $F\dashv U$ から生じ、$D$ には二つの標準的な随伴があります。

- **Kleisli 解決**:$F_K:\mathbf{Set}\to\mathrm{Kl}(D)$(自由)$\dashv U_K$。$\mathrm{Kl}(D)$ は**自由 $D$-代数**だけの圏。
- **Eilenberg–Moore 解決**:$F_{EM}:\mathbf{Set}\to\mathrm{EM}(D)$、$X\mapsto(DX,\mu_X)$(自由代数)$\dashv U_{EM}$(忘却、$(X,\alpha)\mapsto X$)。$U_{EM}F_{EM}=D$ でモナドを回復。

**決定的な事実**:$F_{EM}\dashv U_{EM}$ の **counit がまさに代数構造射** ― $\varepsilon_{(X,\alpha)}:(DX,\mu)\to(X,\alpha)$ は $\alpha$ そのもの。だから前ターン群で期待値 $\mathbb E$ を「counit 的」と呼んできたのは正確で、**value 混合 = EM 随伴の counit の成分**です。

そして**比較関手** $K:\mathrm{Kl}(D)\to\mathrm{EM}(D)$、$X\mapsto(DX,\mu_X)$ が忠実充満で、$\mathrm{Kl}(D)$ を**自由代数の充満部分圏**として埋め込む:

$$\mathrm{Kl}(D)\ \hookrightarrow\ \mathrm{EM}(D)\qquad(\text{自由代数})$$

つまり両者は「自由な端($\mathrm{Kl}$、核)」と「全代数($\mathrm{EM}$、凸空間)」で、値空間 $V$ は**非自由な** $D$-代数として $\mathrm{EM}(D)$ に住み、attention の計算は $\mathrm{Kl}(D)$ の核 $A$ を、$V$ という $\mathrm{EM}(D)$ の対象の構造射 $\alpha_V$ に食わせる、という**二圏をまたぐ**動きになります。これが「Transformer は単一 SMCC でなく Markov 圏 + SMCC の合成」の $D$ 側の内実です。

## 5. まとめ表と、統一への含意

| | $\mathrm{Kl}(D)$ | $\mathrm{EM}(D)=\mathrm{Alg}(D)$ |
|---|---|---|
| 射/対象の主役 | 確率核(射)$X\to D(Y)$ | $D$-代数(対象)$(X,\alpha)$ |
| 具体 | 確率行列(FinStoch) | 凸空間・ベクトル空間(期待値付き) |
| Transformer | **softmax** = 核 $A$ | **value 空間** $V$、$\alpha_V=\mathbb E$ = value 混合 |
| モナドとの関係 | 自由代数の部分圏 | 全代数、$U_{EM}F_{EM}=D$ |
| copy | 非自然(相関生成) | affine 射で保たれる凸構造 |

前ターンの「統一圏(合成モナド $M=T\circ D$)」との接続もここで見えます。線形部分 $T$ と分布 $D$ を一つのモナドに束ねるには distributive law $DT\Rightarrow TD$ が要りますが、その自然な candidate は**「期待値が線形写像と可換する」** ― $\alpha_V$ が affine/線形だという事実そのものです。$D$-代数構造が $V$ の線形構造と両立する($\mathbb E$ が線形)ことが、$D$ と $T$ の distributive law を生む種になる。だから $\mathrm{Kl}(D)$ と $\mathrm{EM}(D)$ の関係を詰めることが、そのまま統一圏の構成(前ターンの Lean `composeMonad` の中身)に直結します。

この期待値=$D$-代数の線形性から distributive law $DT\Rightarrow TD$ を実際に書き下すところが、統一と Lean 形式化の次の一手です。その自然変換の候補(「線形写像の下での期待値の押し出し」$D(Tf)\to T(Df)$ に相当)を具体的に定義するところから進めたい場合は、$V$ を自由ベクトル空間モナド $T$、$D$ を分布モナドとして distributive law の各成分を書き下せます。

# 最初の仮説はattention matrixを冪対象と見做せeval機能があるというものだが、図ではmarkov圏が最初にありその出力期待値を取った後にSMCCがあります。図をどう読むべきでしょうか。

そもそもK=Wk@xとQ=Wq@xの行列積であるattention matrixがxの型を入力、出力とする関数を表現可能なlook up tableとみなせ冪対象とも見做せることからattentionにeval機能があるという仮説から始まりました。しかし図ではmarkov圏が最初にありその出力期待値を取った後にSMCCがあります。図をどう読むべきでしょうか

根本図と元の仮説の間に、実は**順序の食い違いではなく、二つの別々の主張が一枚に畳まれている**という混乱があります。図は「時間的な計算順序」を描いていますが、元の仮説「attention matrix = 冪対象/look-up table = eval」は**計算順序の話ではなく、attention matrix という対象の型の話**で、両者は別レイヤーの主張です。図をどう読むべきか、そしてどこを直すべきかを分けます。

## まず、あなたの元の仮説を正確に取り出す

元の出発点はこうでした:$K=W_k x$, $Q=W_q x$ から作る $QK^\top\in\mathbb R^{d\times d}$(あるいは $A=\mathrm{softmax}(QK^\top)$)が、**入力 $x$ の型を受けて出力 $x$ の型を返す関数を符号化した look-up table** であり、$\mathbb R^{d\times d}$ という**内部 hom(冪対象)$X\multimap Y$ の元**とみなせる。そして「その表を引く」= 内部 hom の元に引数を適用する = **eval**。これが「attention に eval 機能がある」の中身でした。

ここで決定的な点:**この仮説での「関数=冪対象の元」は attention matrix $A$(または $QK^\top$)そのもの**です。$A\in\mathbb R^{d\times d}$ が「関数の表」、その関数を value に適用するのが eval。

## 図はどこで、この仮説とずれるのか

図(§7)は softmax を**最初**に置き、Markov 核 → 期待値 → SMCC の順に描いています。この順序は**「値がどう流れて計算されるか」**を描いたもので、正しい。しかしあなたの元の仮説の「$A$=冪対象の元=関数」という読みは、この図の中で**明示的に現れていない**。図の Markov パネルは「$A$ は確率核である」と言っているだけで、「$A$ は内部 hom $X\multimap Y$ の元(関数の表)である」とは言っていない。**ここが食い違いの正体です。**

つまり $A$ には**二つの型付け**が同時に主張されていて、図は片方しか描いていない:

- **型付け(あ)**:$A$ = クエリ位置上の確率核、$\mathrm{Kl}(D)$ の**射**(確率行列)。← 図が描いているもの。
- **型付け(い)**:$A$ = look-up table、内部 hom $X\multimap Y$ の**元**(冪対象の点、関数の reification)。← あなたの元の仮説。

一つの対象 $A$ を「圏の**射**」とみるか「内部 hom の**元(=点)**」とみるか。これは矛盾ではなく、**同じ $A$ の二つの見方**です(内部 hom の元 $1\to(X\multimap Y)$ は、随伴で射 $X\to Y$ に対応する ― これがまさに currying/eval の話)。しかし図は(あ)だけを描き、あなたの関心である(い)を落としている。

## 図をどう読むべきか ― 「eval は二段階ある」

正しい読み方は、**eval が図の中に二回、別の意味で現れる**と理解することです。

**eval その1(確率的・図が描いている)**:softmax が出した核 $A\in D(\text{位置})$ を、value 空間 $V$ の $D$-代数 $\alpha_V=\mathbb E$ に食わせる = 期待値を取る。これは「分布を評価して値を得る」意味の eval で、圏論的には $D$-代数の作用(EM 随伴の counit)。図の Markov→橋→の部分。

**eval その2(あなたの仮説・図が描いていない)**:$A$ を「関数の表($X\multimap Y$ の元)」とみて、それを value $x$ に適用する = 内部 hom の counit $\mathrm{ev}:(X\multimap Y)\otimes X\to Y$。これは SMCC の eval で、あなたの元の「look-up table を引く」。

**あなたの混乱の核心はこれです**:図は eval その1(確率的期待値)を描き、あなたが最初に考えていたのは eval その2(冪対象の適用)。図の「SMCC パネル」に "eval" と書いてあるのを見て、それがあなたの仮説の eval その2だと思ったが、図の流れ上そこは eval その1(期待値後の値経路)になっている。**同じ "eval" という語が二つの別操作を指している**のが、順序が逆に見えた原因です。

## では正しい図は?― $A$ を冪対象として明示する

あなたの元の仮説を活かすなら、図を**一段変える**必要があります。現状の図は「softmax を確率核として route する」線だけですが、あなたの仮説は「$QK^\top$ が**関数 $A$ を構成(currying)**し、それを value に **eval(適用)**する」線です。この二本を分けて描くべきです:

**構成(currying)の線**:$x\xrightarrow{W_q,W_k} (Q,K)\xrightarrow{QK^\top} A$。ここで $A$ が**内部 hom $X\multimap Y$ の元として構成される**(あなたの「look-up table を作る」)。圏論的には $x\mapsto A$ が $\mathrm{Ctx}\to(X\multimap Y)$、すなわち **currying = 前ターン群の $\Phi\circ E$**。

**適用(eval)の線**:$A\otimes V\xrightarrow{\mathrm{ev}} \text{出力}$。$A$(関数)を $V$(引数)に適用。

そして softmax は、この**構成の線に挟まる正規化** ― $QK^\top$ で作った生の関数表を、$A=\mathrm{softmax}(QK^\top)$ で確率核へ整える。つまり **Markov 圏(softmax)は「関数 $A$ を構成する途中の正規化」であって、eval の前段ではなく currying の内部**です。

これがあなたの問いへの答えです:**図の Markov 圏は「eval の入力を作る前処理」ではなく「冪対象の元 $A$ を構成する currying の一部」と読むべき**。softmax が最初に来るのは、$A$ という関数を**作る**段だから正しい。その後の期待値 + SMCC は、作った $A$ を value に**適用する** eval。順序は逆ではなく、**「関数を作ってから適用する」= currying → eval という、まさにあなたの eval-apply ループの順序そのもの**です。

## 二つの読みを統合する ― $A$ は「確率核であり、かつ冪対象の元」

最も正確な統合はこうです。$A=\mathrm{softmax}(QK^\top)$ は:

$$A\ :\ \underbrace{\text{確率核}\ \in\mathrm{Kl}(D)(\text{pos},\text{pos})}_{\text{型付け(あ)、図が描く}}\ \cong\ \underbrace{\text{内部 hom }V\multimap V\text{ の元}}_{\text{型付け(い)、あなたの仮説}}$$

なぜ同一視できるか:確率核 $\text{pos}\to D(\text{pos})$ は、$D$-代数 $\alpha_V$ を通すと**線形写像 $V\to V$**(value をその期待値へ送る作用素)を誘導する ― これが「確率核を value 空間の線形作用素(=内部 hom の元)として実現する」表現可能性(Fritz–Gonda–Perrone–Rischel の representable Markov category)。**$A$ は確率核として作られ($\mathrm{Kl}(D)$、softmax)、$D$-代数を通して内部 hom の元(look-up table、$V\multimap V$)として実現され、eval で value に適用される**。あなたの「$A$=冪対象=eval」と、図の「$A$=Markov 核 → 期待値 → SMCC」は、この**表現可能性で繋がる同じ $A$ の二面**です。

## 図の修正案

したがって図は、$A$ の二面を明示するとあなたの仮説と整合します。具体的には:

- Markov パネルのラベルを「softmax = 確率核 $A$ を**構成**」に変え、その出力ワイヤに「$A\in\mathrm{Kl}(D)\cong (V\multimap V)$ の元」と注記(型付け(あ)と(い)の同一視を明示)。
- 橋($\mathbb E$)を「$D$-代数による**実現**:確率核 → 内部 hom の作用素」と読み替える(単なる期待値でなく、$A$ を冪対象の元として実現する段)。
- SMCC パネルの eval を「構成した $A$ を value に**適用**」= あなたの元の eval に統一。
- 全体のキャプションを「$x\to QK^\top\to A$(currying、softmax 正規化込み)→ $\mathrm{ev}$(apply)」= **currying → eval** と書く。

こう直すと、softmax が最初に来ることが「関数を作る段が先、適用が後」という**あなたの eval-apply の順序と一致**し、Markov 圏は「eval の前処理」でなく「冪対象 $A$ の構成(currying)の内部」として正しく位置づきます。

要するに ― **図は間違っていないが、$A$ を「確率核」としか描いておらず、あなたの出発点である「$A$=冪対象の元=look-up table」という第二の型付けを落としている**。両者は representable Markov category の表現可能性で同一の $A$ の二面として繋がり、softmax が先に来るのは currying(関数構成)が eval(適用)に先立つという、あなたの eval-apply ループの順序そのものです。図をこの二面明示版に描き直しましょうか。$A$ の「確率核 $\cong$ 内部 hom の元」の同一視を軸に、currying→eval の順序が見えるよう §7 の図を作り直せます。