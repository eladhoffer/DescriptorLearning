require 'nn'
require 'eladtools'
require 'BatchHingeEmbeddingCriterion'
require 'BatchPairwiseDistance'
local opt = opt or {type = 'cuda'}
local InputMaps = 3
local InputWidth = 32
local InputHeight = 32

local FeatMaps = {32,64}
local KernelSize = {7,3}
local ConvStride = {1,1}
local PoolSize = {2,2}
local PoolStride= PoolSize
local Outputs = 128
local Margin = Outputs
local TValue = 0
local TReplace = 0.000001 

local SizeMap = {InputWidth}
for i=2, #FeatMaps+1 do
    SizeMap[i] = math.floor(math.floor((SizeMap[i-1] - KernelSize[i-1] + 1) / ConvStride[i-1]) / PoolStride[i-1])
end

if opt.type == 'cuda' then
    require 'cunn'
end



local model = nn.Sequential()

---------------Layer - Convolution + Max Pooling------------------
LayerNum = 1
model:add(nn.SpatialConvolutionMM(InputMaps, FeatMaps[LayerNum], KernelSize[LayerNum], KernelSize[LayerNum], ConvStride[LayerNum], ConvStride[LayerNum]))
--model:add(nn.Threshold(TValue, TReplace))
model:add(nn.Tanh())
model:add(nn.SpatialMaxPooling(PoolSize[LayerNum], PoolSize[LayerNum], PoolStride[LayerNum], PoolStride[LayerNum]))


---------------Layer - Convolution + Max Pooling------------------
LayerNum = 2
model:add(nn.SpatialConvolutionMM(FeatMaps[LayerNum-1], FeatMaps[LayerNum], KernelSize[LayerNum], KernelSize[LayerNum], ConvStride[LayerNum], ConvStride[LayerNum]))
--model:add(nn.Threshold(TValue, TReplace))
model:add(nn.Tanh())
model:add(nn.SpatialMaxPooling(PoolSize[LayerNum], PoolSize[LayerNum], PoolStride[LayerNum], PoolStride[LayerNum]))

---------------Layer - Fully connected classifier ------------------
LayerNum = 3
--model:add(nn.Reshape(SizeMap[LayerNum]*SizeMap[LayerNum]*FeatMaps[LayerNum-1]))
model:add(nn.SpatialConvolutionMM(FeatMaps[LayerNum-1], Outputs, SizeMap[LayerNum], SizeMap[LayerNum]))
--model:add(nn.Linear(SizeMap[LayerNum]*SizeMap[LayerNum]*FeatMaps[LayerNum-1], Outputs))
model:add(nn.Tanh())
model:add(nn.Reshape(Outputs))
--model:add(nn.SoftMax())
--model:add(nn.Threshold())
--local loss = nn.L2EmbeddingCriterion(100)
if opt.type == 'cuda' then
    --model = CPU2CUDA(model)
    model = model:cuda()
    --loss:cuda()
end


-- Retrieve parameters and gradients:
-- this extracts and flattens all the trainable parameters of the mode
-- into a 1-dim vector
local w,dE_dw = model:getParameters()
local model_copy = model:clone('weight','bias','gradWeight','gradBias')
local parallel = nn.ParallelTable()
parallel:add(model)
parallel:add(model_copy)

local EmbeddingModel = nn.Sequential()
EmbeddingModel:add(parallel)
EmbeddingModel:add(nn.BatchPairwiseDistance(1))

local loss = nn.BatchHingeEmbeddingCriterion(Margin)
if opt.type == 'cuda' then
    EmbeddingModel:cuda()
    loss = loss:cuda()
end

return {
    model = EmbeddingModel,
    SubModel = model,
    weight = w,
    grad = dE_dw,
    InputSize = {InputMaps, InputWidth, InputHeight},
    loss = loss
}
