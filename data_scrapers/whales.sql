WITH top_10k_wallets AS (
  SELECT 
    address,
    eth_balance
  FROM `bigquery-public-data.crypto_ethereum.balances` 
  ORDER BY 2 DESC
  LIMIT 10000
)

SELECT
  DATE(block_timestamp) AS block_date,
  COUNT(DISTINCT from_address) AS unique_whales,
  COUNT(DISTINCT transactions.hash) AS transactions_number,
  SUM(value) AS total_value_in_wei,
  SAFE_DIVIDE(SUM(value), POW(10, 18)) AS total_value_in_eth
FROM `bigquery-public-data.crypto_ethereum.transactions` AS transactions
INNER JOIN top_10k_wallets
  ON top_10k_wallets.address = transactions.from_address 
WHERE DATE(block_timestamp) >= "2017-09-01"
  AND value > 0
GROUP BY 1
ORDER BY 1

