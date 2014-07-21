
require 'torch'   -- torch
require 'xlua'    -- xlua provides useful tools, like progress bars
require 'optim'   -- an optimization package, for online and batch methods

----------------------------------------------------------------------
-- Model + Loss:
local t = require 'model'
local model = t.model
local marginThreshold = t.loss.margin/2
local InputSize = t.InputSize
----------------------------------------------------------------------
print '==> defining some tools'



print '==> allocating minibatch memory'

local x1 = torch.Tensor(opt.batchSize, InputSize[1], InputSize[2], InputSize[3])
local x2 = torch.Tensor(opt.batchSize, InputSize[1], InputSize[2], InputSize[3])
local similar = torch.Tensor(opt.batchSize)


if opt.type == 'cuda' then 
    x1 = x1:cuda()
    x2 = x2:cuda()
end

----------------------------------------------------------------------
print '==> defining testing procedure'


local function test(testData)

    -- local vars
    local time = sys.clock()
    local numS = 0
    local numN = 0
    local DistanceS = 0
    local DistanceN = 0


    local class_error = 0

    for t = 1,testData.MatchList:size(1),opt.batchSize do

        -- disp progress
        xlua.progress(t, testData.MatchList:size(1))
        collectgarbage()

        -- batch fits?
        if (t + opt.batchSize - 1) > testData.MatchList:size(1) then
            break
        end

        -- create mini batch
        local idx = 1
        for i = t,t+opt.batchSize-1 do
            x1[idx] = testData.Patches[testData.MatchList[i][1]]
            x2[idx] = testData.Patches[testData.MatchList[i][2]]
            if testData.ID[testData.MatchList[i][1]] == testData.ID[testData.MatchList[i][2]] then
                similar[idx] = 1
            else
                similar[idx] = -1
            end

            idx = idx + 1
        end

        local y = model:forward({x1,x2})

        for p=1,opt.batchSize do

            if similar[p] == -1 then
                DistanceN = DistanceN + y[p]
                numN = numN +1
            else
                numS = numS +1
                DistanceS = DistanceS + y[p]
            end
            local pred
            if y[p] > marginThreshold then 
                pred = -1
            else
                pred = 1
            end
            class_error = class_error + math.abs(pred-similar[p])/2
        end


    end

    -- time taken
    time = sys.clock() - time
    time = time / testData.MatchList:size(1)
    print("\n==> time to test 1 sample = " .. (time*1000) .. 'ms')

    DistanceS = DistanceS / numS
    DistanceN = DistanceN / numN
    class_error = class_error / (numN+numS)
    marginThreshold = (DistanceN-DistanceS)/2 + DistanceS
    print('(TEST) Distance of similar patches = ' .. DistanceS)
    print('(TEST) Distance of dissimilar patches = ' .. DistanceN)
    print('(TEST) Modified Decision Threshold = ' .. marginThreshold)
    print('(TEST) Class Error = ' .. class_error*100 ..'%')

    return 1-class_error
end

-- Export:
return test

