{-# LANGUAGE OverloadedStrings #-}

module MiniLispEval
  ( Expr(..)
  , Closure(..)
  , Env
  , evalExpr
  , eval
  , modN
  ) where

import qualified Data.Map.Strict as M
import           Data.Map.Strict (Map)
import           Debug.Trace (trace)

--------------------------------------------------------------------------------
-- AST / Values
--------------------------------------------------------------------------------
data Expr
  = EInt Int
  | EBool Bool
  | ENum Double              -- Python の "/" で float が出るので保持
  | EStr String              -- 変数/シンボル
  | EList [Expr]             -- S式 (Python の list)
  | EListVal [Expr]          -- ("list", elems)
  | EClosure Closure
  deriving (Eq, Show)

data Closure = Closure
  { cParams :: [String]
  , cBody   :: Expr
  , cEnv    :: Env
  } deriving (Eq, Show)

type Env = Map String Expr

-- Python の MOD
modN :: Int
modN = 7

--------------------------------------------------------------------------------
-- Top-level
--------------------------------------------------------------------------------

evalExpr :: Expr -> Maybe Env -> (Expr, Int)
evalExpr expr mEnv = eval expr (maybe M.empty id mEnv)

--------------------------------------------------------------------------------
-- Helpers: env / predicates
--------------------------------------------------------------------------------

isClosure :: Expr -> Bool
isClosure (EClosure _) = True
isClosure _            = False

copyEnv :: Env -> Env
copyEnv = id

-- make_new_env(env, *[(name,val)])
extendEnvPairs :: Env -> [(String, Expr)] -> Env
extendEnvPairs base pairs = foldl (\e (k,v) -> M.insert k v e) base pairs

-- make_new_env(env, params, arg_vals)
extendEnvZip :: Env -> [String] -> [Expr] -> Env
extendEnvZip base ps vs = extendEnvPairs base (zip ps vs)

--------------------------------------------------------------------------------
-- Arg evaluation (Python: _evalarg / _evalargs)
--------------------------------------------------------------------------------

data ArgVals = One Expr | Many [Expr]
  deriving (Eq, Show)

-- Python: _evalarg(arg, env, steps)
evalArg :: Expr -> Env -> Int -> (Expr, Int)
evalArg arg env steps0 =
  let (v, st) = eval arg env
  in (v, steps0 + st)

-- Python: __evalargs(args:list, env, steps)
evalArgsList :: [Expr] -> Env -> Int -> ([Expr], Int)
evalArgsList args env steps0 = go [] steps0 args
  where
    go acc steps []     = (reverse acc, steps)
    go acc steps (a:as) =
      let (v, st) = eval a env
      in go (v:acc) (steps + st) as

-- Python: _evalargs(args, env, steps)
--   if isinstance(args, list): __evalargs
--   else: _evalarg
evalArgs :: Expr -> Env -> Int -> (ArgVals, Int)
evalArgs args env steps0 =
  case args of
    EList xs ->
      let (vals, steps1) = evalArgsList xs env steps0
      in (Many vals, steps1)
    _ ->
      let (v, steps1) = evalArg args env steps0
      in (One v, steps1)

--------------------------------------------------------------------------------
-- Core eval (Python: _eval)
--------------------------------------------------------------------------------

ops :: [String]
ops = ["+", "-", "*", "/"]

cmps :: [String]
cmps = ["==", "<", ">", "<=", ">="]

eval :: Expr -> Env -> (Expr, Int)
eval expr env =
  case expr of
    -- atoms
    EInt _  -> (expr, 0)
    EBool _ -> (expr, 0)
    ENum _  -> (expr, 0)
    EClosure _ -> (expr, 0)

    -- variable
    EStr s ->
      case M.lookup s env of
        Just v  -> (v, 0)
        Nothing -> (expr, 0)

    -- ("list", elems)
    EListVal xs ->
      let (vals, steps1) = evalArgsList xs env 0
      in (EListVal vals, steps1)

    -- S式
    EList xs ->
      case xs of
        [] -> (expr, 0)
        (h:_) ->
          case h of
            EStr headSym ->
              let _dbg = trace ("head: " <> headSym <> " expr: " <> show expr) ()
              in
                if headSym `elem` ops
                  then evalOp headSym (tail xs) env
                else if headSym `elem` cmps
                  then evalCmp headSym (tail xs) env
                else dispatchSpecial headSym expr env
            _ ->
              evalApp expr env

    -- その他はそのまま
    _ -> (expr, 0)

dispatchSpecial :: String -> Expr -> Env -> (Expr, Int)
dispatchSpecial headSym expr env =
  case headSym of
    "if"      -> evalIf expr env
    "fn"      -> evalFn expr env
    "let"     -> evalLet expr env
    "compose" -> evalCompose expr env
    "partial" -> evalPartial expr env
    "map"     -> evalMap expr env
    "filter"  -> evalFilter expr env
    "reduce"  -> evalReduce expr env
    "cons"    -> evalCons expr env
    "first"   -> evalFirst expr env
    "rest"    -> evalRest expr env
    "append"  -> evalAppend expr env
    "len"     -> evalLen expr env
    _         -> evalApp expr env

--------------------------------------------------------------------------------
-- Special forms (Python versions)
--------------------------------------------------------------------------------

-- Python: _eval_if(expr, env)    # (if cond then else)
evalIf :: Expr -> Env -> (Expr, Int)
evalIf expr env =
  case expr of
    EList [EStr "if", condE, thenE, elseE] ->
      let (condV, steps0) = evalArg condE env 0
      in case condV of
           EBool b ->
             let steps1 = steps0 + 1
                 target = if b then thenE else elseE
                 (v, st2) = eval target env
             in (v, steps1 + st2)
           _ ->
             (EList [EStr "if", condV, thenE, elseE], steps0)
    _ -> (expr, 0)

-- Python: _eval_fn(expr, env)    # (fn [params...] body) -> closure
evalFn :: Expr -> Env -> (Expr, Int)
evalFn expr env =
  case expr of
    EList [EStr "fn", paramsE, body] ->
      let raw =
            case paramsE of
              EListVal xs -> Just xs
              _           -> Just paramsE  -- Python は tuple/list を切替; ここは後段で検査
          paramsMaybe =
            case raw of
              Just (EList ps) -> allStr ps >>= \ss -> Just ss
              Just (EListVal ps) -> allStr ps >>= \ss -> Just ss
              Just other ->
                -- paramsE が単体で来た場合は失敗扱いに寄せる（Python と同様に厳しめ）
                case other of
                  EList ps    -> allStr ps
                  EListVal ps -> allStr ps
                  _           -> Nothing
              Nothing -> Nothing
      in case paramsMaybe of
           Nothing -> (expr, 0)
           Just ps ->
             let clo = Closure { cParams = ps, cBody = body, cEnv = copyEnv env }
                 _dbg = trace ("closure " <> show clo) ()
             in (EClosure clo, 1)
    _ -> (expr, 0)
  where
    allStr :: [Expr] -> Maybe [String]
    allStr = traverse (\e -> case e of EStr s -> Just s; _ -> Nothing)

-- Python: _eval_let(expr, env)
-- (let ((x e1) (y e2) ...) body)
-- - RHS は外側 env で評価してから並列束縛
evalLet :: Expr -> Env -> (Expr, Int)
evalLet expr env =
  case expr of
    EList [EStr "let", bindingsE, body] ->
      case bindingsE of
        EList binds -> doLet binds
        _           -> (expr, 0)
      where
        doLet :: [Expr] -> (Expr, Int)
        doLet binds =
          case traverse parseBind binds of
            Nothing -> (expr, 0)
            Just namedRhss ->
              let (evaluatedPairs, stepsSum) = evalAllRhs namedRhss env
                  newEnv = extendEnvPairs env evaluatedPairs
                  (v, stepsFinal) = evalArg body newEnv stepsSum
              in (v, stepsFinal)

        parseBind :: Expr -> Maybe (String, Expr)
        parseBind b =
          case b of
            EList [EStr name, rhs] -> Just (name, rhs)
            _                      -> Nothing

        evalAllRhs :: [(String, Expr)] -> Env -> ([(String, Expr)], Int)
        evalAllRhs pairs env0 = go [] 0 pairs
          where
            go acc steps [] = (reverse acc, steps)
            go acc steps ((name, rhs):xs) =
              let (v, steps1) = evalArg rhs env0 0
              in go ((name, v):acc) (steps + steps1) xs
    _ -> (expr, 0)

--------------------------------------------------------------------------------
-- Operators / comparisons (Python _eval_op / _eval_cmp)
--------------------------------------------------------------------------------

evalOp :: String -> [Expr] -> Env -> (Expr, Int)
evalOp op args env =
  let (vals, steps0) = evalArgsList args env 0
  in if not (null vals) && all isInt vals
       then
         let steps1 = steps0 + 1
         in case op of
              "+" -> (EInt (sum (map asInt vals) `mod` modN), steps1)
              "-" -> case vals of
                       [v] ->
                         let x = asInt v
                         in (EInt ((negate x) `mod` modN), steps1)
                       [v1,v2] ->
                         (EInt ((asInt v1 - asInt v2) `mod` modN), steps1)
                       (v1:rest) ->
                         (EInt ((asInt v1 - sum (map asInt rest)) `mod` modN), steps1)
                       _ -> (EList (EStr op : vals), steps0)
              "*" -> (EInt (product (map asInt vals) `mod` modN), steps1)
              "/" ->
                -- Python は 0 除算で部分式を残す
                let xs = map asInt vals
                in case xs of
                     []     -> (EList (EStr op : vals), steps0)
                     (x:ys) ->
                       if any (==0) ys
                         then (EList (EStr op : vals), steps0)
                         else
                           let resD = foldl (\acc y -> acc / fromIntegral y) (fromIntegral x :: Double) ys
                               resM = modDouble resD (fromIntegral modN)
                           in (ENum resM, steps1)
              _ -> (EList (EStr op : vals), steps0)
       else
         (EList (EStr op : vals), steps0)
  where
    isInt (EInt _) = True
    isInt _        = False
    asInt (EInt x) = x
    asInt _        = error "asInt: non-int (guarded)"

    -- Python の x % MOD に相当する近似（Double）
    modDouble :: Double -> Double -> Double
    modDouble x m =
      let k = fromIntegral (floor (x / m) :: Integer)
      in x - k * m

evalCmp :: String -> [Expr] -> Env -> (Expr, Int)
evalCmp op args env =
  let (vals, steps0) = evalArgsList args env 0
  in if length vals >= 2 && all isIntOrBool vals
       then
         let steps1 = steps0 + 1
             ok = case op of
                    "==" -> allAdjacentEq vals
                    "<"  -> allAdjacentCmp (<) vals
                    ">"  -> allAdjacentCmp (>) vals
                    "<=" -> allAdjacentCmp (<=) vals
                    ">=" -> allAdjacentCmp (>=) vals
                    _    -> False
         in (EBool ok, steps1)
       else
         (EList (EStr op : vals), steps0)
  where
    isIntOrBool (EInt _)  = True
    isIntOrBool (EBool _) = True
    isIntOrBool _         = False

    toOrd :: Expr -> Either Int Bool
    toOrd (EInt x)  = Left x
    toOrd (EBool b) = Right b
    toOrd _         = error "toOrd: guarded"

    allAdjacentEq :: [Expr] -> Bool
    allAdjacentEq xs = and $ zipWith (==) xs (tail xs)

    -- Python の mixed (int,bool) をそのまま比較しているので、
    -- Haskell でも「型が違う場合は False」に寄せる。
    allAdjacentCmp :: (forall a. Ord a => a -> a -> Bool) -> [Expr] -> Bool
    allAdjacentCmp f xs = and $ zipWith cmp xs (tail xs)
      where
        cmp a b =
          case (toOrd a, toOrd b) of
            (Left x, Left y)   -> f x y
            (Right x, Right y) -> f x y
            _                  -> False

--------------------------------------------------------------------------------
-- Application / compose / partial / list ops
--------------------------------------------------------------------------------

-- Python: _eval_app(expr, env)  # (f a1 a2 ...)
evalApp :: Expr -> Env -> (Expr, Int)
evalApp expr env =
  case expr of
    EList (fE:argEs) ->
      let (fVal, steps1)   = evalArg fE env 0
          (argValsE, steps2) = evalArgsList argEs env steps1
      in case fVal of
           EClosure clo ->
             let params0 = cParams clo
                 body    = cBody clo
                 fEnv    = cEnv clo
             in
               -- Python: if arg_vals is closure then params = arg_vals["params"]
               case argValsE of
                 [EClosure clo2] ->
                   let params = cParams clo2
                   in applyClosure params body fEnv [] steps2
                 _ ->
                   if length argValsE /= length params0
                     then (EList (fVal : argValsE), steps2)
                     else
                       let newEnv = extendEnvZip fEnv params0 argValsE
                           steps3 = steps2 + 1
                       in
                         -- Python は _evalargs(body,new_env,steps)
                         -- （body が list なら複数評価、単体なら単体評価）
                         case evalArgs body newEnv steps3 of
                           (One v,  st) -> (v, st)
                           (Many vs, st) -> (EList vs, st)
           _ ->
             (EList (fVal : argValsE), steps2)
    _ -> (expr, 0)
  where
    applyClosure :: [String] -> Expr -> Env -> [Expr] -> Int -> (Expr, Int)
    applyClosure params body fEnv args steps =
      let newEnv = extendEnvZip fEnv params args
          steps1 = steps + 1
      in case evalArgs body newEnv steps1 of
           (One v,  st) -> (v, st)
           (Many vs, st) -> (EList vs, st)

-- Python: _eval_compose(expr, env) # (compose f g)  ~> (fn [x] (f (g x)))
evalCompose :: Expr -> Env -> (Expr, Int)
evalCompose expr env =
  case expr of
    EList [EStr "compose", fE, gE] ->
      let param = "_c_arg"
          composedAst =
            EList
              [ EStr "fn"
              , EListVal [EStr param]
              , EList [ fE, EList [ gE, EStr param ] ]
              ]
          (v, st) = eval composedAst env
      in (v, st + 1)
    _ -> (expr, 0)

-- Python: _eval_partial(expr, env)   # (partial f a1 a2 ...)
evalPartial :: Expr -> Env -> (Expr, Int)
evalPartial expr env =
  case expr of
    EList (EStr "partial" : fE : argEs) | not (null argEs) ->
      let (fVal, steps1) = evalArg fE env 0
          (argVals, steps2) = evalArgsList argEs env steps1
      in case fVal of
           EClosure clo ->
             let params = cParams clo
                 body   = cBody clo
                 fEnv   = cEnv clo
             in if length argVals >= length params
                  then
                    let newEnv = extendEnvZip fEnv params argVals
                        (v, steps3) = evalArg body newEnv (steps2 + 1)
                    in (v, steps3)
                  else
                    let remaining = drop (length argVals) params
                        boundEnv   = extendEnvZip fEnv params argVals
                        clo' = Closure remaining body boundEnv
                    in (EClosure clo', steps2 + 1)
           _ ->
             (EList (EStr "partial" : fVal : argVals), steps2)
    _ -> (expr, 0)

-- Python: _eval_map(expr, env)    # (map f list) -> ("list", ...)
evalMap :: Expr -> Env -> (Expr, Int)
evalMap expr env =
  case expr of
    EList [EStr "map", fE, listE] ->
      let (fVal, steps1) = evalArg fE env 0
          (lstV, steps2) = evalArg listE env steps1
      in case (fVal, lstV) of
           (EClosure _, EListVal elems) ->
             let (resElems, steps3) = go [] steps2 elems
             in (EListVal resElems, steps3 + 1)
           _ ->
             (EList [EStr "map", fVal, lstV], steps2)
    _ -> (expr, 0)
  where
    go acc steps [] = (reverse acc, steps)
    go acc steps (e:es) =
      let (applied, st) = evalApp (EList [fValPlaceholder, e]) env
          -- ここは Python では [f_val, elem] を _eval_app へ
          -- f_val は外側スコープなので、実装上はクロージャを捕捉して使う必要がある
      in go (applied:acc) (steps + st) es

    -- この placeholder は使われない（下の改良版 evalMap2 を推奨）
    fValPlaceholder = EStr "__internal_error__"

-- 上の evalMap は fVal を go に渡していないので、正しい版をこちらに置きます。
-- （Python 相当の挙動）
evalMap2 :: Expr -> Env -> (Expr, Int)
evalMap2 expr env =
  case expr of
    EList [EStr "map", fE, listE] ->
      let (fVal, steps1) = evalArg fE env 0
          (lstV, steps2) = evalArg listE env steps1
      in case (fVal, lstV) of
           (EClosure _, EListVal elems) ->
             let (resElems, steps3) = go fVal [] steps2 elems
             in (EListVal resElems, steps3 + 1)
           _ ->
             (EList [EStr "map", fVal, lstV], steps2)
    _ -> (expr, 0)
  where
    go _ acc steps [] = (reverse acc, steps)
    go fVal acc steps (e:es) =
      let (applied, st) = evalApp (EList [fVal, e]) env
      in go fVal (applied:acc) (steps + st) es

-- Python: _eval_filter(expr, env)    # (filter pred list) -> ("list", ...)
evalFilter :: Expr -> Env -> (Expr, Int)
evalFilter expr env =
  case expr of
    EList [EStr "filter", pE, listE] ->
      let (pVal, steps1) = evalArg pE env 0
          (lstV, steps2) = evalArg listE env steps1
      in case (pVal, lstV) of
           (EClosure _, EListVal elems) ->
             let (res, steps3) = go pVal [] steps2 elems
             in (EListVal res, steps3 + 1)
           _ ->
             (EList [EStr "filter", pVal, lstV], steps2)
    _ -> (expr, 0)
  where
    go _ acc steps [] = (reverse acc, steps)
    go pVal acc steps (e:es) =
      let (condV, st) = evalApp (EList [pVal, e]) env
          steps' = steps + st
      in case condV of
           EBool True  -> go pVal (e:acc) steps' es
           EBool False -> go pVal acc     steps' es
           _           -> go pVal (e:acc) steps' es  -- 判定できない場合は残す

-- Python: _eval_reduce(expr, env)  # (reduce f list init) -> expr
evalReduce :: Expr -> Env -> (Expr, Int)
evalReduce expr env =
  case expr of
    EList [EStr "reduce", fE, listE, initE] ->
      let (fVal, steps1) = evalArg fE env 0
          (lstV, steps2) = evalArg listE env steps1
          (acc0, steps3) = evalArg initE env steps2
      in case (fVal, lstV) of
           (EClosure _, EListVal elems) ->
             let (accF, steps4) = foldl step (acc0, steps3) elems
             in (accF, steps4 + 1)
           _ ->
             (EList [EStr "reduce", fVal, lstV, acc0], steps3)
    _ -> (expr, 0)
  where
    step (acc, steps) e =
      let (acc', st) = evalApp (EList [fValPlaceholder, acc, e]) env
      in (acc', steps + st)

    -- placeholder 同様、正しい版は evalReduce2 を推奨
    fValPlaceholder = EStr "__internal_error__"

evalReduce2 :: Expr -> Env -> (Expr, Int)
evalReduce2 expr env =
  case expr of
    EList [EStr "reduce", fE, listE, initE] ->
      let (fVal, steps1) = evalArg fE env 0
          (lstV, steps2) = evalArg listE env steps1
          (acc0, steps3) = evalArg initE env steps2
      in case (fVal, lstV) of
           (EClosure _, EListVal elems) ->
             let (accF, steps4) = foldl (step fVal) (acc0, steps3) elems
             in (accF, steps4 + 1)
           _ ->
             (EList [EStr "reduce", fVal, lstV, acc0], steps3)
    _ -> (expr, 0)
  where
    step fVal (acc, steps) e =
      let (acc', st) = evalApp (EList [fVal, acc, e]) env
      in (acc', steps + st)

-- Python: _eval_cons(expr, env)
evalCons :: Expr -> Env -> (Expr, Int)
evalCons expr env =
  case expr of
    EList [EStr "cons", headE, listE] ->
      let (hV, steps1)  = evalArg headE env 0
          (lstV, steps2) = evalArg listE env steps1
      in case lstV of
           EListVal xs -> (EListVal (hV:xs), steps2 + 1)
           _           -> (EList [EStr "cons", hV, lstV], steps2)
    _ -> (expr, 0)

-- Python: _eval_first(expr, env)
evalFirst :: Expr -> Env -> (Expr, Int)
evalFirst expr env =
  case expr of
    EList [EStr "first", listE] ->
      let (lstV, steps1) = evalArg listE env 0
      in case lstV of
           EListVal (x:_) -> (x, steps1 + 1)
           _              -> (EList [EStr "first", lstV], steps1)
    _ -> (expr, 0)

-- Python: _eval_rest(expr, env)
evalRest :: Expr -> Env -> (Expr, Int)
evalRest expr env =
  case expr of
    EList [EStr "rest", listE] ->
      let (lstV, steps1) = evalArg listE env 0
      in case lstV of
           EListVal (_:xs) -> (EListVal xs, steps1 + 1)
           _               -> (EList [EStr "rest", lstV], steps1)
    _ -> (expr, 0)

-- Python: _eval_append(expr, env)
evalAppend :: Expr -> Env -> (Expr, Int)
evalAppend expr env =
  case expr of
    EList [EStr "append", l1E, l2E] ->
      let (l1V, steps1) = evalArg l1E env 0
          (l2V, steps2) = evalArg l2E env steps1
      in case (l1V, l2V) of
           (EListVal a, EListVal b) -> (EListVal (a <> b), steps2 + 1)
           _ -> (EList [EStr "append", l1V, l2V], steps2)
    _ -> (expr, 0)

-- Python: _eval_len(expr, env)
-- ※ Python 版は lst_v を _evalargs(list_e,...) で評価し、
--   その後 isinstance(lst_v, list) and lst_v[0] == "list" をチェックしていますが、
--   あなたの他関数の表現（("list", ...)）と整合しません。
--   ここでは実質意図に合わせ、EListVal を長さ計算対象にします。
evalLen :: Expr -> Env -> (Expr, Int)
evalLen expr env =
  case expr of
    EList [EStr "len", listE] ->
      let (v, steps1) = evalArg listE env 0
      in case v of
           EListVal xs -> (EInt (length xs), steps1 + 1)
           _           -> (EList [EStr "len", v], steps1)
    _ -> (expr, 0)

--------------------------------------------------------------------------------
-- NOTE: map/reduce の「正しい版」への差し替え
--------------------------------------------------------------------------------

-- 上で evalMap/evalReduce は Python のコード構造をそのまま写している関係で
-- クロージャ fVal を内部ループに渡す必要があります。
-- 実運用では dispatchSpecial の "map"/"reduce" を evalMap2/evalReduce2 に差し替えてください。
--
-- 例:
--   "map"    -> evalMap2 expr env
--   "reduce" -> evalReduce2 expr env
--
-- Python の挙動としてもこちらが自然です。
