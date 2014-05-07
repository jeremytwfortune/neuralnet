# Encoding: utf-8

##
# Author: Jeremy Fortune
# Contact: jeremytwfortune@gmail.com
# Purpose: 
# Prerequisites:
# Returns: 
# Known Issues: 
#

load 'Neuron.rb'
require 'thread'

###############################################################################

##
# Taken from t-a-w.blogspot.com.
def Exception.ignoring_exceptions
  begin
    yield
  rescue Exception => e
    STDERR.puts e.message
  end
end

module Enumerable
  def in_parallel(n)
    todo = Queue.new
    ts = (1..n).map{
      Thread.new{
        while x = todo.deq
          Exception.ignoring_exceptions{ yield(x[0]) } 
        end
      }
    }
    each{|x| todo << [x]}
    n.times{ todo << nil }
    ts.each{|t| t.join}
  end
end

###############################################################################

class BackPropNet
  attr_accessor :range, :predictors, :predicted
  attr_reader :ilayer, :olayer, :hlayer, :output, :mse
  
  def initialize(predictors, predicted, hnum = 10)
    @predictors = predictors
    @predicted = predicted
    @range = (0..@predicted.size-1)
    @mse = []

    @ilayer = Array.new(predictors[0].size) { Neuron.new }
    @ilayer.each { |i| i.weights = [1] }
    @hlayer = Array.new(hnum) { Neuron.new(@ilayer.size) }
    @olayer = Array.new(predicted[0].size) { Neuron.new(@hlayer.size) }

    @output = Array.new(@olayer.size)
  end

  ##
  # Sets +output+ and feeds data from the input layer through the network.
  def feed(data)
    input = []
    hidden = []
    @ilayer.each_index do |i|
      @ilayer[i].inputs = [data[i]]
      input[i] = @ilayer[i].activate
    end
    @hlayer.each_index do |h| 
      @hlayer[h].inputs = input
      hidden[h] = @hlayer[h].activate
    end
    @olayer.each_index do |o| 
      @olayer[o].inputs = hidden
      @output[o] = @olayer[o].activate
    end
    @output
  end
  
  ##
  # Trains the network using input data.
  def train(data)
    @olayer.each_index do |o|
      # Compute output gradient.
      @olayer[o].error = {value: data[o], exact: false}
      # Train the output layer.
      @olayer[o].train 
    end

    # Train hidden layer.
    @hlayer.each_index do |h|
      wsum = 0
      @olayer.each{ |o| wsum = wsum + (o.weights[h] * o.gradient) }
      @hlayer[h].error = wsum
      @hlayer[h].train
    end
  end
  
  ##
  # Runs a single epoch over +predictors+ within +range+.
  def epoch
    sqerror = 0
    @range.each do |row|
      self.feed(@predictors[row])
      self.train(@predicted[row])
      rowerror = 0
      @olayer.each{ |o| rowerror = rowerror + o.error**2 }
      sqerror = sqerror + rowerror
    end
    
    @mse.push(sqerror / @range.count)
    @mse.last
  end
  
  ##
  # For multiple output layers, vote for a state.
  def vote
    elected = 0
    @output.each_index do |o|
      elected = @output[o] > @output[elected] ? o : elected
    end
    elected
  end    
end


      