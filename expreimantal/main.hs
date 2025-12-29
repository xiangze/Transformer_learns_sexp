{-# LANGUAGE OverloadedStrings #-}

module Main where

import Control.Monad (forM_, replicateM)
import Control.Monad.Random (evalRand, mkStdGen)
import Data.Maybe (fromMaybe)
import System.IO (withFile, IOMode(WriteMode), hPutStrLn)
import Options.Applicative

-- ここはあなたが既に持っているモジュールに合わせて import を調整してください。
-- Parser / Generator / AST
import qualified SexpRandMR as G   -- MonadRandom 版 generator: Expr/Kind/genExpr/renderSexp/randomTypedSexpN...
import qualified MiniSexp   as E   -- evaluator: Expr/Closure/Env/evalExpr など
import qualified SexpParser as P   -- megaparsec parser: parseSexp :: String -> Either String Expr

--------------------------------------------------------------------------------
-- sexpr_to_str: 内部表現を S式文字列へ
-- Python は closure を "(closure fn [params] body)" と表示していたので、それに合わせる。
--------------------------------------------------------------------------------
sexprToStr :: E.Expr -> String
sexprToStr expr =
  case expr of
    E.EInt n       -> show n
    E.EBool True   -> "True"
    E.EBool False  -> "False"
    E.EStr s       -> s
    E.EListVal xs  -> "[" <> unwords (map sexprToStr xs) <> "]"
    E.EList xs     -> "(" <> unwords (map sexprToStr xs) <> ")"
    E.EClosure clo ->
      let params = unwords (E.cParams clo)
          body   = sexprToStr (E.cBody clo)
      in "(closure fn [" <> params <> "] " <> body <> ")"

--------------------------------------------------------------------------------
-- totaleval: parse(tokenize(expr_str)) -> eval_expr(...)
-- Haskell は tokenize を分けず parseSexp で直接 parse
--------------------------------------------------------------------------------
totalEval :: String -> Either String (E.Expr, Int)
totalEval s = do
  ast <- P.parseSexp s
  let (v, steps) = E.evalExpr ast Nothing
  pure (v, steps)

--------------------------------------------------------------------------------
-- reduce_and_show
--------------------------------------------------------------------------------
reduceAndShow :: String -> IO ()
reduceAndShow exprStr =
  case P.parseSexp exprStr of
    Left err -> do
      putStrLn ("元の式:    " <> exprStr)
      putStrLn ("Parse error:\n" <> err)
      putStrLn (replicate 60 '-')
    Right ast -> do
      let (value, steps) = E.evalExpr ast Nothing
      putStrLn ("元の式:    " <> exprStr)
      putStrLn ("AST:       " <> sexprToStr ast)
      putStrLn ("簡約結果:  " <> sexprToStr value)
      putStrLn ("ステップ数: " <> show steps)
      putStrLn (replicate 60 '-')

--------------------------------------------------------------------------------
-- eval_demo
--------------------------------------------------------------------------------

evalDemo :: IO ()
evalDemo = do
  let samples =
        [ "(+ 1 2 3)"
        , "(if True (+ 1 2) 5)"
        , "((fn [x] (+ x 1)) 3)"
        , "(map (fn [x] (+ x 1)) [1 2 3])"
        , "(filter (fn [x] (> x 2)) [1 2 3 4])"
        , "(reduce (fn [a b] (+ a b)) [1 2 3] 0)"
        , "(len [1 2 3 4])"
        , "(compose (fn [x] (+ x 1)) (fn [y] (* y 2)))"
        ]
  putStrLn "demo"
  forM_ samples reduceAndShow

--------------------------------------------------------------------------------
-- gen_and_eval: generator で式を作り、評価して (exprStr, valueStr, steps)
-- Python は want_kind を文字列で渡していたので、Haskell では Kind に変換する
--------------------------------------------------------------------------------

kindFromString :: String -> G.Kind
kindFromString s =
  case s of
    "int"     -> G.KInt
    "bool"    -> G.KBool
    "list"    -> G.KList
    "closure" -> G.KClosure
    "any"     -> G.KAny
    _         -> G.KInt

genAndEval
  :: Int      -- num_exprs
  :: Int      -- max_depth
  :: String   -- want_kind ("int"/"bool"/...)
  :: Int      -- seed
  -> [(String, String, Int)]
genAndEval numExprs maxDepth wantKindStr seed =
  let want = kindFromString wantKindStr
      g0   = mkStdGen seed
      -- generator は G.Expr だが、eval は E.Expr を使うので、同一 AST 型に揃える必要がある。
      -- ここでは「同一型で統合済み」という前提で、G.Expr ≡ E.Expr として書く。
      -- もしモジュールが別型なら、変換関数を書いて統一してください。
      xs :: [(String, G.Kind)]
      xs = G.randomTypedSexpN numExprs maxDepth want seed
  in
    [ case totalEval s of
        Left _ -> (s, "<parse/eval error>", 0)
        Right (v, st) -> (s, sexprToStr v, st)
    | (s, _) <- xs
    ]

--------------------------------------------------------------------------------
-- gen_and_eval_print: show なら標準出力、otherwise ならファイルへ
--------------------------------------------------------------------------------
data Params = Params
  { pN          :: Int
  , pMaxDepth   :: Int
  , pSeed       :: Maybe Int
  , pTargetFree :: Maybe Int   -- この版では未実装（Python でも best-effort）
  , pShow       :: Bool
  , pShowShort  :: Bool
  , pValType    :: String
  , pDemo       :: Bool
  } deriving (Show)

genAndEvalPrint :: Params -> FilePath -> IO ()
genAndEvalPrint p baseName = do
  let seed = fromMaybe 42 (pSeed p)
      numExprs = pN p
      maxDepth = pMaxDepth p
      vt = pValType p

      outFile = baseName <> "_n" <> show numExprs <> "_d" <> show maxDepth <> "_kind" <> vt <> ".txt"
      results = genAndEval numExprs maxDepth vt seed
  if pShow p
    then do
      forM_ (zip [0 :: Int ..] results) $ \(i,(exprStr, valStr, steps)) -> do
        putStrLn (show i <> " " <> replicate 40 '-')
        putStrLn ("Generated expr: " <> exprStr)
        if pShowShort p
          then putStrLn ("Evaluated to   : " <> take 120 valStr)
          else putStrLn ("Evaluated to   : " <> valStr)
        putStrLn ("Steps taken    : " <> show steps)
    else
      withFile outFile WriteMode $ \h ->
        forM_ results $ \(exprStr, valStr, steps) ->
          hPutStrLn h (exprStr <> ", " <> valStr <> ", " <> show steps)

--------------------------------------------------------------------------------
-- CLI (argparse 相当): optparse-applicative
--------------------------------------------------------------------------------
paramsParser :: Parser Params
paramsParser =
  Params
    <$> option auto (long "n" <> value 10 <> help "number of expressions")
    <*> option auto (long "max_depth" <> value 4 <> help "maximum nesting depth")
    <*> optional (option auto (long "seed" <> help "random seed"))
    <*> optional (option auto (long "target_free" <> help "require exactly this many distinct free variables (best-effort)"))
    <*> switch (long "show" <> help "show result")
    <*> switch (long "show_short" <> help "show short result")
    <*> strOption (long "valtype" <> value "int" <> help "value type to generate (int,bool,list,closure,any)")
    <*> switch (long "demo" <> help "show demo")

main :: IO ()
main = do
  p <- execParser (info (paramsParser <**> helper) fullDesc)
  if pDemo p
    then evalDemo
    else do
      let results = genAndEval (pN p) (pMaxDepth p) (pValType p) 42
      putStrLn "length,"
      forM_ results $ \(s,_,_) -> putStrLn s
      forM_ results $ \(s,v,st) ->
        putStrLn (show (length s) <> " " <> show (length v) <> " " <> show st <> " steps")
