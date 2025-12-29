{-# LANGUAGE OverloadedStrings #-}

module SexpRand
  ( Expr(..)
  , Kind(..)
  , parseSexp
  , renderSexp
  , randomTypedSexp
  , randomTypedSexpN
  , genExpr
  ) where

import Control.Monad (replicateM)
import Control.Monad.State.Strict (State, evalState, get, put)
import Data.Char (isAlphaNum, isDigit)
import Data.Foldable (toList)
import Data.List (intercalate)
import System.Random (StdGen, mkStdGen, randomR)

import qualified Text.Megaparsec as MP
import Text.Megaparsec (Parsec, (<|>))
import qualified Text.Megaparsec.Char as MPC

--------------------------------------------------------------------------------
-- AST (Python parse の出力に対応)
--------------------------------------------------------------------------------
data Expr
  = EInt Int
  | EBool Bool
  | EStr String
  | EList [Expr]      -- (...) の S式
  | EListVal [Expr]   -- [...] の list literal  (Python parse では ("list", elems))
  deriving (Eq, Show)

data Kind = KInt | KBool | KList | KClosure | KAny
  deriving (Eq, Show)

--------------------------------------------------------------------------------
-- Pretty printer (Expr -> S式文字列)
--------------------------------------------------------------------------------
renderSexp :: Expr -> String
renderSexp e =
  case e of
    EInt n      -> show n
    EBool True  -> "True"
    EBool False -> "False"
    EStr s      -> s
    EList xs    -> "(" <> unwords (map renderSexp xs) <> ")"
    EListVal xs -> "[" <> unwords (map renderSexp xs) <> "]"

--------------------------------------------------------------------------------
-- Parser (あなたの tokenize/parse と同等)
--------------------------------------------------------------------------------
type Parser = Parsec () String

sc :: Parser ()
sc = MPC.space MPC.space1 (MP.skipLineComment ";") MP.empty

lexeme :: Parser a -> Parser a
lexeme p = p <* MP.optional sc

symbol :: String -> Parser String
symbol s = lexeme (MPC.string s)

parseSexp :: String -> Either String Expr
parseSexp input =
  case MP.parse (sc *> pExpr <* sc <* MP.eof) "<sexp>" input of
    Left err -> Left (MP.errorBundlePretty err)
    Right x  -> Right x

pExpr :: Parser Expr
pExpr = pParens  <|> pBrackets  <|> pAtom

pParens :: Parser Expr
pParens = do
  _ <- symbol "("
  xs <- MP.many pExpr
  _ <- symbol ")"
  pure (EList xs)

pBrackets :: Parser Expr
pBrackets = do
  _ <- symbol "["
  xs <- MP.many pExpr
  _ <- symbol "]"
  pure (EListVal xs)

pAtom :: Parser Expr
pAtom = lexeme (pBool <|> pInt <|> pIdentOrOp)

pInt :: Parser Expr
pInt = do
  ds <- MP.some (MP.satisfy isDigit)
  pure (EInt (read ds))

pBool :: Parser Expr
pBool =
      (MPC.string "True"  >> pure (EBool True))
  <|> (MPC.string "False" >> pure (EBool False))

-- identifiers / operators:
-- <= >= ==, single ops + - * / < >, identifiers [A-Za-z_][A-Za-z0-9_]*
pIdentOrOp :: Parser Expr
pIdentOrOp = do
  s <- pTok
  pure (EStr s)

pTok :: Parser String
pTok =
      MP.try (MPC.string "<=")
  <|> MP.try (MPC.string ">=")
  <|> MP.try (MPC.string "==")
  <|> MP.try (MP.singleton <$> MP.oneOf ("+-*/<>" :: String))
  <|> pIdent
  where
    pIdent = do
      h <- MP.satisfy (\c -> c == '_' || ('A' <= c && c <= 'Z') || ('a' <= c && c <= 'z'))
      t <- MP.many (MP.satisfy (\c -> c == '_' || isAlphaNum c))
      pure (h:t)

--------------------------------------------------------------------------------
-- Random S-expression generator (Python gen_* 群の移植)
--------------------------------------------------------------------------------

-- Python 側の定数
modN :: Int
modN = 7

values :: [Expr]
values = map (EInt . read) (map show [0 .. modN - 1])

bools :: [Expr]
bools = [EBool True, EBool False]

baseVars :: [String]
baseVars = ["x","y","z","t"]

ops :: [String]
ops = ["+","-","*","/"]

cmps :: [String]
cmps = ["==","<",">","<=",">="]

-- RNG helpers
randR :: (Int, Int) -> State StdGen Int
randR range = do
  g <- get
  let (x, g') = randomR range g
  put g'
  pure x

randUnit :: State StdGen Double
randUnit = do
  n <- randR (0, 10^9)
  pure (fromIntegral n / 1e9)

choice :: [a] -> State StdGen a
choice xs = do
  i <- randR (0, length xs - 1)
  pure (xs !! i)

-- weighted choice: [(item, weight)]
weightedChoice :: [(a, Int)] -> State StdGen a
weightedChoice items = do
  let total = sum (map snd items)
  r <- randR (1, total)
  pure (pick r items)
  where
    pick _ [] = error "weightedChoice: empty"
    pick k ((a,w):rest)
      | k <= w    = a
      | otherwise = pick (k - w) rest

randomVar :: State StdGen Expr
randomVar = do
  u <- randUnit
  if u < 0.5
    then EStr <$> choice baseVars
    else do
      n <- randR (0, 9)
      pure (EStr ("v" <> show n))
--------------------------------------------------------------------------------
-- gen_terminal(depth, want_kind)
--------------------------------------------------------------------------------
genTerminal :: Int -> Kind -> State StdGen (Expr, Kind)
genTerminal _ want =
  case want of
    KInt -> do
      u <- randUnit
      if u < 0.7
        then (,KInt) <$> choice values
        else do v <- randomVar; pure (v, KInt)
    KBool ->
      (,KBool) <$> choice bools
    KList ->
      pure (EListVal [], KList)  -- 深さ0では空リスト寄せ
    KClosure -> do
      v <- randomVar
      -- (fn [v] v)
      let ast = EList [EStr "fn", EListVal [v], v]
      pure (ast, KClosure)
    KAny -> do
      k <- choice [KInt, KBool, KList, KClosure]
      genTerminal 0 k
--------------------------------------------------------------------------------
-- gen_list_literal(depth): list ::= [expr ...]
--------------------------------------------------------------------------------
genListLiteral :: Int -> State StdGen (Expr, Kind)
genListLiteral depth = do
  n <- if depth <= 0 then randR (0,2) else randR (2,5)
  elems <- replicateM n (fst <$> genExpr (depth - 1) KAny)
  pure (EListVal elems, KList)

-- gen_op / gen_cmp
genOp :: Int -> State StdGen (Expr, Kind)
genOp depth = do
  op <- EStr <$> choice ops
  arity <- randR (2,4)
  args <- replicateM arity (fst <$> genExpr (depth - 1) KInt)
  pure (EList (op : args), KInt)

genCmp :: Int -> State StdGen (Expr, Kind)
genCmp depth = do
  op <- EStr <$> choice cmps
  arity <- randR (2,3)
  args <- replicateM arity (fst <$> genExpr (depth - 1) KInt)
  pure (EList (op : args), KBool)

-- gen_if
genIf :: Int -> Kind -> State StdGen (Expr, Kind)
genIf depth want = do
  cond <- fst <$> genExpr (depth - 1) KBool
  (t,_) <- genExpr (depth - 1) want
  (f,_) <- genExpr (depth - 1) want
  pure (EList [EStr "if", cond, t, f], want)

-- gen_fn
genFn :: Int -> State StdGen (Expr, Kind)
genFn depth = do
  numParams <- randR (1,3)
  params <- replicateM numParams randomVar
  (body,_) <- genExpr (depth - 1) KAny
  pure (EList [EStr "fn", EListVal params, body], KClosure)

-- gen_app
genApp :: Int -> Kind -> State StdGen (Expr, Kind)
genApp depth _want = do
  (fun,_) <- genExpr (depth - 1) KClosure
  arity <- randR (1,3)
  args <- replicateM arity (fst <$> genExpr (depth - 1) KAny)
  pure (EList (fun : args), KAny)

--------------------------------------------------------------------------------
-- compose / partial / map / filter / reduce / cons / first / rest / append / len
--------------------------------------------------------------------------------
genCompose :: Int -> State StdGen (Expr, Kind)
genCompose depth = do
  f1 <- fst <$> genExpr (depth - 1) KClosure
  f2 <- fst <$> genExpr (depth - 1) KClosure
  pure (EList [EStr "compose", f1, f2], KClosure)

genPartial :: Int -> State StdGen (Expr, Kind)
genPartial depth = do
  (fun,_) <- genExpr (depth - 1) KClosure
  arity <- randR (1,3)
  args <- replicateM arity (fst <$> genExpr (depth - 1) KAny)
  pure (EList (EStr "partial" : fun : args), KClosure)

genList :: Int -> State StdGen (Expr, Kind)
genList depth
  | depth <= 0 = genListLiteral depth
  | otherwise = do
      kind <- weightedChoice
        [ ("literal", 2)
        , ("map",     4)
        , ("filter",  3)
        , ("cons",    3)
        , ("rest",    2)
        , ("append",  3)
        ]
      case kind of
        "literal" -> genListLiteral depth
        "map"     -> genMap depth
        "filter"  -> genFilter depth
        "cons"    -> genCons depth
        "rest"    -> genRest depth
        "append"  -> genAppend depth
        _         -> genListLiteral depth

genMap :: Int -> State StdGen (Expr, Kind)
genMap depth = do
  (f,_) <- genExpr (depth - 1) KClosure
  (lst,_) <- genList (depth - 1)
  pure (EList [EStr "map", f, lst], KList)

genFilter :: Int -> State StdGen (Expr, Kind)
genFilter depth = do
  (p,_) <- genExpr (depth - 1) KClosure
  (lst,_) <- genList (depth - 1)
  pure (EList [EStr "filter", p, lst], KList)

genReduce :: Int -> Kind -> State StdGen (Expr, Kind)
genReduce depth want = do
  (f,_) <- genExpr (depth - 1) KClosure
  (lst,_) <- genList (depth - 1)
  (initE,_) <- genExpr (depth - 1) want
  pure (EList [EStr "reduce", f, lst, initE], want)

genCons :: Int -> State StdGen (Expr, Kind)
genCons depth = do
  (e,_) <- genExpr (depth - 1) KAny
  (lst,_) <- genList (depth - 1)
  pure (EList [EStr "cons", e, lst], KList)

genFirst :: Int -> State StdGen (Expr, Kind)
genFirst depth = do
  (lst,_) <- genList (depth - 1)
  pure (EList [EStr "first", lst], KAny)

genRest :: Int -> State StdGen (Expr, Kind)
genRest depth = do
  (lst,_) <- genList (depth - 1)
  pure (EList [EStr "rest", lst], KList)

genAppend :: Int -> State StdGen (Expr, Kind)
genAppend depth = do
  (l1,_) <- genList (depth - 1)
  (l2,_) <- genList (depth - 1)
  pure (EList [EStr "append", l1, l2], KList)

genLen :: Int -> State StdGen (Expr, Kind)
genLen depth = do
  (lst,_) <- genList (depth - 1)
  pure (EList [EStr "len", lst], KInt)

--  gen_let: (let ((x e1) (y e2) ...) body)
genLet :: Int -> State StdGen (Expr, Kind)
genLet depth = do
  names <- replicateM 3 randomVar
  binds <- forM names $ \nm ->
    case nm of
      EStr s -> do
        (rhs,_) <- genExpr (depth - 1) KAny
        pure (EList [EStr s, rhs])
      _ -> error "randomVar must return EStr"
  (body,_) <- genExpr (depth - 1) KAny
  pure (EList [EStr "let", EList binds, body], KAny)
  where
    forM = flip mapM

-- gen_expr main (Python gen_expr の候補/重み構造を移植)
genExpr :: Int -> Kind -> State StdGen (Expr, Kind)
genExpr depth want
  | depth <= 0 = genTerminal depth want
  | otherwise = do
      let candidates = case want of
            KInt ->
              [ ("value_terminal", 1)
              , ("op", 4)
              , ("len", 3)
              , ("reduce", 4)
              , ("if_int", 2)
              , ("app", 2)
              ]
            KBool ->
              [ ("bool_terminal", 1)
              , ("cmp", 5)
              , ("if_bool", 2)
              , ("app", 1)
              ]
            KList ->
              [ ("list", 5)
              , ("if_list", 2)
              , ("app", 1)
              ]
            KClosure ->
              [ ("fn", 4)
              , ("compose", 3)
              , ("partial", 3)
              , ("if_closure", 1)
              ]
            KAny ->
              [ ("op", 2)
              , ("cmp", 1)
              , ("list", 2)
              , ("fn", 2)
              , ("let", 2)
              , ("app", 4)
              , ("compose", 2)
              , ("partial", 2)
              , ("if_any", 2)
              , ("first", 2)
              , ("reduce", 3)
              ]
      k <- weightedChoice candidates
      case k of
        "value_terminal" -> genTerminal depth KInt
        "bool_terminal"  -> genTerminal depth KBool
        "op"             -> genOp depth
        "cmp"            -> genCmp depth
        "len"            -> genLen depth
        "fn"             -> genFn depth
        "compose"        -> genCompose depth
        "partial"        -> genPartial depth
        "list"           -> genList depth
        "map"            -> genMap depth
        "filter"         -> genFilter depth
        "reduce"         -> genReduce depth (if want == KAny then KAny else want)
        "first"          -> genFirst depth
        "let"            -> genLet depth
        "app"            -> genApp depth want
        "if_int"         -> genIf depth KInt
        "if_bool"        -> genIf depth KBool
        "if_list"        -> genIf depth KList
        "if_closure"     -> genIf depth KClosure
        "if_any"         -> genIf depth KAny
        _                -> genTerminal depth want

--------------------------------------------------------------------------------
-- Public wrappers (Python random_typed_sexp / random_typed_sexp_n)
--------------------------------------------------------------------------------

randomTypedSexp :: Int -> Kind -> Maybe Int -> (String, Kind)
randomTypedSexp maxDepth want seed =
  let g = maybe (mkStdGen 0) mkStdGen seed
      (e, k) = evalState (genExpr maxDepth want) g
  in (renderSexp e, k)

randomTypedSexpN :: Int -> Int -> Kind -> Maybe Int -> [(String, Kind)]
randomTypedSexpN n maxDepth want seed =
  let g = maybe (mkStdGen 0) mkStdGen seed
      genOne = genExpr maxDepth want
      xs = evalState (replicateM n genOne) g
  in map (\(e,k) -> (renderSexp e, k)) xs
