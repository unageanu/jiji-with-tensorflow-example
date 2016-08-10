require 'httpclient'
require 'json'

client = HTTPClient.new

result = client.post("http://localhost:5001/api/estimator", {
  body: JSON.generate({
    macd: -0.23609241853516494,
    macd_signal: -0.22064377650158612,
    macd_difference: -0.015448642033578819,
    rsi_9: 22.037218413320243,
    rsi_14: 43.62292051756002,
    slope_25: -0.018881907692311618,
    slope_50: 0.0009629714285715573,
    slope_75: 0.0317762583214802,
    ma_25_estrangement: -0.8370332266232502,
    ma_50_estrangement: -0.7993143975514959,
    ma_75_estrangement: -0.7615668631781956,
    stochastics_k: 0,
    stochastics_d: 34.75298126064736,
    stochastics_sd: 34.117660420215785,
    fast_stochastics: -34.75298126064736,
    slow_stochastics: 0.6353208404315751,
    sell_or_buy: "sell"
  }),
  header: {
    'Content-Type' => 'application/json'
  }
})
p result.body
