require 'pl'
require 'torch'
require 'eladtools'
require 'trepl'

print '==> processing options'

opt = lapp[[
   -r,--learningRate       (default 0.1)         learning rate
   -d,--learningRateDecay  (default 1e-7)        learning rate decay (in # samples)
   -w,--weightDecay        (default 1e-4)        L2 penalty on the weights
   -m,--momentum           (default 0.5)         momentum
   -d,--dropout            (default 0)           dropout amount
   -b,--batchSize          (default 128)         batch size
   -t,--threads            (default 8)           number of threads
   -p,--type               (default float)       float or cuda
   -i,--devid              (default 1)           device ID (if using CUDA)
   -s,--size               (default extra)       dataset: small or full or extra
   -o,--save               (default results)     save directory
]]

-- nb of threads and fixed seed (for repeatable experiments)
torch.setnumthreads(opt.threads)
torch.manualSeed(1)
torch.setdefaulttensortype('torch.FloatTensor')

-- type:
if opt.type == 'cuda' then
   print('==> switching to CUDA')
   require 'cunn'
   cutorch.setDevice(opt.devid)
   print('==> using GPU #' .. cutorch.getDevice())
   print(cutorch.getDeviceProperties(opt.devid))
end

----------------------------------------------------------------------
print '==> load modules'

t = require 'model'

local data  = require 'data'
local train = require 'train'
local test  = require 'test'
local earlystopper = EarlyStop(5)
----------------------------------------------------------------------
print '==> training!'

repeat
   train(data.trainData)
   TestSuccessRate = test(data.testData)
   earlystopper:Update(TestSuccessRate)
until earlystopper:Stop()


earlystopper:PrintStatus()
