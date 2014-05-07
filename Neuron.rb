# Encoding: utf-8

##
# Author: Jeremy Fortune
# Contact: jeremytwfortune@gmail.com
# Purpose: 
# Prerequisites:
# Returns: 
# Known Issues: 
#

###############################################################################

class Neuron
  attr_accessor :bias, :weights, :rate, :inputs, :gradient, :output, :momentum
  attr_reader :size, :error, :lastchange
  
  def initialize(size = 1)
    @weights = Array.new(size, 0)
    @inputs = Array.new(size, 0)
    @bias = 1
    @size = size
    @output = 0
    @gradient = 0
    @error = 0
    @lastchange = 0
    @rate = 0.5
    @momentum = 0.0
    self.mutate
  end

  ##
  # Resets the size of the +weights+ array. If the size is greater, adds zero
  # elements until size is met. Othewise, removes weights from +weights+.
  def size=(size)
    dif = size - @size
    dif < 0 ? @weights.pop(-dif) : @weights.push(Array.new(dif, 0)).flatten!
    dif < 0 ? @inputs.pop(-dif) : @inputs.push(Array.new(dif, 0)).flatten!
    @size = size
  end
  
  ##
  # Randomly assigns weights to the +weights+ array.
  def mutate
    @weights = (1..@size).map { Random.rand(-0.5..0.5) }
  end
  
  ##
  # Return the weighted sum of inputs. Assumes inputs are contained in the 
  # neuron if not provided.
  def weightedsum
    ws = 0
    @inputs.each_index { |i| ws = ws + @inputs[i]*@weights[i] }
    ws
  end
  
  ##
  # Apply the sigmoid acivation function and return the result.
  def activate
    @output = (1.0 / (1.0 + Math.exp(-1 * self.weightedsum - @bias)))
    @output
  end
  
  ##
  # Determine +error+ and +gradient+. Value is used exactly if exact is
  # true in a hash, otherwise the value is compared against +output+.
  def error=(e)
    if e.is_a?(Hash) 
      value = e[:value]
      exact = e[:exact]
    else
      value = e
      exact = true
    end
    
    @error = exact ? value : (value - @output)
    @gradient = @output * (1 - @output) * @error
    @error
  end
  
  ##
  # Alters the +weights+ array according to +rate+ and +gradient+.
  def train
    @weights.each_index do |i| 
      # alter weight and apply momentum
      @weights[i] = @weights[i] + (@rate * inputs[i] * @gradient)
      @weights[i] = @weights[i] + @momentum * @lastchange
      
      @lastchange = @rate * inputs[i] * @gradient
    end
    @weights
  end
  
end