import Text.ParserCombinators.Parsec (parseFromFile)
import Data.CSV
import Numeric.LinearAlgebra
import qualified Data.Attoparsec.ByteString as B
import qualified Data.ByteString as BS

-- load vector from csv
load_vector :: String -> IO (Vector R)
load_vector path = do
    m <- load_matrix path
    return $ (toColumns m) !! 0

-- load matrix from csv
load_matrix :: String -> IO (Matrix R)
load_matrix path = do
    csv <- parseFromFile csvFile path
    case csv of
        Right d -> return $ fromLists $ map (map read) d
        Left  _ -> error "parsing file failed!"

-- load labels from binary data
labelsBinParser :: B.Parser [Int]
labelsBinParser = do
    B.take 8
    (map fromIntegral) <$> (B.many' B.anyWord8)

load_labels :: String -> IO [Int]
load_labels path = do
    dat <- BS.readFile path 
    let res = B.parseOnly labelsBinParser dat
     in case res of
         Right x -> return x
         Left  e -> error e

-- load images from binary data
imgPart :: B.Parser [Int]
imgPart = do
    img <- B.count 784 B.anyWord8
    return $ map fromIntegral img

imagesBinParser :: B.Parser [[Int]]
imagesBinParser = do
    B.take 16
    B.many' imgPart

load_images :: String -> IO [[Int]]
load_images path = do
    dat <- BS.readFile path 
    let res = B.parseOnly imagesBinParser dat
     in case res of
         Right x -> return x
         Left  e -> error e

sigmoid :: Vector R -> Vector R
sigmoid x = 1 / (1 + (exp $ negate x))

softmax :: Vector R -> Vector R
softmax xs =
    let s = sumElements (cmap exp xs)
     in cmap (\x -> exp x / s) xs

type Dataset = (Matrix R, Matrix R, Matrix R, Vector R, Vector R, Vector R)

predict :: Dataset -> Vector R -> Vector R
predict (w1, w2, w3, b1, b2, b3) x =
    let a1 = (x <# w1) + b1
        z1 = sigmoid a1
        a2 = (z1 <# w2) + b2
        z2 = sigmoid a2
        a3 = (z2 <# w3) + b3
     in softmax a3

predictImages :: Dataset -> [[Int]] -> [Int] -> Int
predictImages ds images labels = sum $ map (\(img, lab) ->
        let p = maxIndex $ predict ds (fromList (map (\i -> (fromIntegral i) / 255) img))
         in if p == lab then 1 else 0
    ) $ zip images labels

main :: IO ()
main = do
    w1 <- load_matrix "sample_weight_W1.csv"
    w2 <- load_matrix "sample_weight_W2.csv"
    w3 <- load_matrix "sample_weight_W3.csv"
    b1 <- load_vector "sample_weight_b1.csv"
    b2 <- load_vector "sample_weight_b2.csv"
    b3 <- load_vector "sample_weight_b3.csv"
    images <- load_images "../dataset/t10k-images-idx3-ubyte"
    label  <- load_labels  "../dataset/t10k-labels-idx1-ubyte"
    print $ predictImages (w1, w2, w3, b1, b2, b3) images label