# Encoding: utf-8

##
# Author: Jeremy Fortune
# Contact: jeremytwfortune@gmail.com
# Purpose: 
# Prerequisites:
# Returns: 
# Known Issues: 
#

require 'csv'

load 'BackPropNet.rb'

###############################################################################

letter = CSV.read('./testing/letter-recognition.data', options = {
  converters: :integer
})

data = CSV.read('./testing/letter-recognition.data', options = {
  converters: :integer,
})
# data.shift

# Build a map
map = []
letter.each { |row| map.push(row[0]) }
map = map.uniq

# Map output values
ans = Array.new(data.size) { Array.new(map.size, 0) }
data.each_index do |row|
  ans[row][map.find_index(data[row][0])] = 1
  data[row].shift
end

bp = BackPropNet.new(data, ans, ((data[0].size + ans[0].size) * 0.66).ceil)

i = 0
j = 500
bp.range = (i..j)
mse = 1
epochs = 0

while (epochs < 3000) do
  epochs = epochs + 1
  if j < (data.size * 0.66).ceil
    i = i + 10
    j = j + 10
  else
    i = 0
    j = 500
  end
  bp.range = (i..j)
  mse = bp.epoch
  print "\r                                                              "
  print "\r#{epochs}: #{mse}"
end

CSV.open('./testing/mse.csv','wb') do |csv|
  bp.mse.each{ |mse| csv << [mse] }
end

generr = []
((data.size * 0.66).ceil..(data.size - 1)).each do |row|
  bp.feed(data[row])
  clust = bp.output
  generr.push([clust, letter[row][0]].flatten)
end

CSV.open('./testing/generr.csv','wb') do |csv|
  generr.each{ |err| csv << err }
end
