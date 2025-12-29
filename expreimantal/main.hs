import SexpRand

main :: IO ()
main = do
  let (s,k) = randomTypedSexp 5 KInt (Just 123)
  putStrLn (show k <> " => " <> s)

  -- パースして AST を確認
  case parseSexp s of
    Left err -> putStrLn err
    Right ast -> print ast

  -- 複数生成
  mapM_ (\(x,kk) -> putStrLn (show kk <> " => " <> x))
        (randomTypedSexpN 10 5 KAny (Just 42))
