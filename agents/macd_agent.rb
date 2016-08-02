

require 'jiji/model/agents/agent'
require 'date'

# TensorFlowと連携してトレードするエージェントのサンプル
class MACDAgent

  include Jiji::Model::Agents::Agent

  def self.description
    <<-STR
TensorFlowと連携してトレードするエージェントのサンプル
      STR
  end

  def self.property_infos
    [
      Property.new('exec_mode', '動作モード("collect" or "trade")', "collect")
    ]
  end

  def post_create
    @calculator = SignalCalculator.new(broker)
    @cross = Cross.new
    @mode  = @exec_mode == 'trade' ?  TradeMode.new : CollectMode.new

    @graph = graph_factory.create('MACD',
      :line, :last, ['#FF9999', '#557777'])
  end

  # 次のレートを受け取る
  def next_tick(tick)
    date = tick.timestamp.to_date
    return if @current_date.nil? || @current_date == date
    @current_date = date
    p date

    signal = @calculator.next_tick(tick)
    @cross.next_data(signal[:macd], signal[:macd_signal])

    @graph << [signal[:macd], signal[:macd_signal]]
    do_trade(signal)
  end

  def do_trade(signal)
    if @cross.cross_up?
      buy(signal)
    elsif @cross.cross_down?
      sell(signal)
    end
  end

  def buy(signal)
    close_exist_positions
    result = broker.buy(:USDJPY, @mode.calculate_units(signal))
    @current_position = broker.positions[result.trade_opened.internal_id]
    @current_signal = signal
  end

  def sell(signal)
    close_exist_positions
    result = broker.sell(:USDJPY, @mode.calculate_units(signal))
    @current_position = broker.positions[result.trade_opened.internal_id]
    @current_signal = signal
  end

  def close_exist_positions
    return unless @current_position
    @current_position.close
    @mode.after_position_closed( @current_signal, @current_position )
    @current_position = nil
    @current_signal = nil
  end

  # データ収集モード
  class CollectMode
    # トレードの数量は10000固定
    def calculate_units(signal)
      10000
    end
    # ポジションが閉じられたら、トレード結果とシグナルをDBに登録する
    def after_position_closed( signal, position )
      TradeAndSignals.create_from( signal, position).save
    end
  end

  # 取引モード
  class TradeMode
    # 最適な数量をtensorflowに問い合わせる
    def calculate_units(signal)
      # TODO
    end
    def after_position_closed( signal, position )
      # do nothing.
    end
  end
end


# ストキャスティクス
# https://www.nomura.co.jp/learn/chart/stochastics.html
class Stochastics < Signals::RangeSignal

  # コンストラクタ
  # k_range:: %Kの集計期間
  # d_range:: %Dの集計期間
  # sd_range:: %SDの集計期間
  def initialize(k_range = 5, d_range = 3, sd_range=3)
    super([k_range, d_range].max)
    @k_range = k_range
    @d_range = d_range
    @sd      = MovingAverage.new(3)
  end

  def calculate(data) #:nodoc:
    k_data   = data[@k_range * -1..-1]
    min    = BigDecimal.new(k_data.min,  10)
    max    = BigDecimal.new(k_data.max,  10)
    k  = calculate_k(k_data, max, min)
    d  = calculate_d(data, max, min)
    sd = @sd.next_data(d)
    { k:k, d:d, sd:sd }
  end

  def calculate_k(k_data, max, min)
    latest = BigDecimal.new(k_data.last, 10)
    ((latest - min) / (max - min) * 100).to_f
  end

  def calculate_d(data, max, min)
    d_data = data[@d_range * -1..-1]
    ((d_data.map {|v| v - min}.reduce(:+)) / ((max - min) * @d_range) * 100).to_f
  end

end


# トレード結果とその時の各種指標。
# MongoDBに格納してTensorFlowの学習データにする
class TradeAndSignals

  include Mongoid::Document

  store_in collection: 'tensorflow_example_signals'

  field :macd,               type: Float
  field :macd_signal,        type: Float
  field :macd_difference,    type: Float # macd - macd_signal

  field :rsi_9,              type: Float
  field :rsi_14,             type: Float

  field :slope_25,           type: Float # 25日移動平均線の傾き
  field :slope_50,           type: Float # 50日移動平均線の傾き
  field :slope_75,           type: Float # 75日移動平均線の傾き

  field :ma_25_estrangement, type: Float # 25日移動平均からの乖離率
  field :ma_50_estrangement, type: Float
  field :ma_75_estrangement, type: Float

  field :stochastics_k,      type: Float
  field :stochastics_d,      type: Float
  field :stochastics_sd,     type: Float
  field :fast_stochastics,   type: Float # stochastics_k - stochastics_d
  field :slow_stochastics,   type: Float # stochastics_d - stochastics_sd

  field :profit_or_loss,     type: Float
  field :sell_or_buy,        type: Symbol
  field :entered_at,         type: Time
  field :exited_at,          type: Time

  def self.create_from( signal_data, position )
    TradeAndSignals.new do |ts|
      signal_data.each do |pair|
        ts.send( "#{pair[0]}=".to_sym, pair[1] )
      end
      ts.profit_or_loss = position.profit_or_loss
      ts.sell_or_buy    = position.sell_or_buy
      ts.entered_at     = position.entered_at
      ts.exited_at      = position.exited_at
    end
  end
end

# シグナルを計算するクラス
class SignalCalculator

  def initialize(broker)
    @broker = broker
  end

  def next_tick(tick)
    prepare_signals(tick) unless @macd
    calculate_signals(tick[:USDJPY])
  end

  def calculate_signals(tick)
    price = tick.bid
    macd = @macd.next_data(price)
    ma25 = @ma25.next_data(price)
    ma50 = @ma25.next_data(price)
    ma75 = @ma25.next_data(price)
    st   = @st.next_data(price)
    {
      macd: macd ? macd[:macd] : nil,
      macd_signal: macd ? macd[:signal] : nil,
      macd_difference: macd ? macd[:macd] - macd[:signal] : nil,
      rsi_9:  @rsi9.next_data(price),
      rsi_14: @rsi14.next_data(price),
      slope_25: ma25 ? @ma25v.next_data(ma25) : nil,
      slope_50: ma50 ? @ma50v.next_data(ma50) : nil,
      slope_75: ma75 ? @ma75v.next_data(ma75) : nil,
      ma_25_estrangement: ma25 ? calculate_estrangement(price, ma25) : nil,
      ma_50_estrangement: ma50 ? calculate_estrangement(price, ma50) : nil,
      ma_75_estrangement: ma75 ? calculate_estrangement(price, ma75) : nil,
      stochastics_k: st ? st[:k] : nil,
      stochastics_d: st ? st[:d] : nil,
      stochastics_sd: st ? st[:sd] : nil,
      fast_stochastics: st ? st[:k] - st[:d] : nil,
      slow_stochastics: st && st[:sd] ? st[:d] - st[:sd] : nil
    }
  end

  def prepare_signals(tick)
    create_signals
    retrieve_rates(tick.timestamp).each do |rate|
      calculate_signals(rate.close)
    end
  end

  def create_signals
    @macd  = Signals::MACD.new
    @ma25  = Signals::MovingAverage.new(25)
    @ma50  = Signals::MovingAverage.new(50)
    @ma75  = Signals::MovingAverage.new(75)
    @ma25v = Signals::Vector.new(25)
    @ma50v = Signals::Vector.new(50)
    @ma75v = Signals::Vector.new(75)
    @rsi9  = Signals::RSI.new(9)
    @rsi14 = Signals::RSI.new(14)
    @st    = Stochastics.new
  end

  def retrieve_rates(time)
    @broker.retrieve_rates(:USDJPY, :one_day, time - 60*60*24*80, time )
  end

  def calculate_estrangement(price, ma)
    ((BigDecimal.new(price, 10) - ma) / ma * 100).to_f
  end

end
