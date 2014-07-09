
require 'torch'   -- torch
require 'xlua'    -- xlua provides useful tools, like progress bars
require 'optim'   -- an optimization package, for online and batch methods

----------------------------------------------------------------------
-- Model + Loss:
local t = require 'model'
local model = t.model
local w, dE_dw = t.weight, t.grad
local loss = t.loss
local InputSize = t.InputSize
----------------------------------------------------------------------
print '==> defining some tools'



----------------------------------------------------------------------
print '==> configuring optimizer'

local optimState = {
    learningRate = opt.learningRate,
    momentum = opt.momentum,
    weightDecay = opt.weightDecay,
    learningRateDecay = opt.learningRateDecay
}

----------------------------------------------------------------------
print '==> allocating minibatch memory'

local x1 = torch.Tensor(opt.batchSize, InputSize[1], InputSize[2], InputSize[3])
local x2 = torch.Tensor(opt.batchSize, InputSize[1], InputSize[2], InputSize[3])
local similar = torch.Tensor(opt.batchSize)


if opt.type == 'cuda' then 
    x1 = x1:cuda()
    x2 = x2:cuda()
end

----------------------------------------------------------------------
print '==> defining training procedure'

local epoch

local function train(trainData)

    -- epoch tracker
    epoch = epoch or 1
    -- local vars
    local time = sys.clock()
    local numS = 0
    local numN = 0
    local DistanceS = 0
    local DistanceN = 0

    -- shuffle at each epoch
    local shuffle1 = torch.randperm(trainData.Patches:size(1))
    local shuffle2 = torch.randperm(trainData.Patches:size(1))

    -- do one epoch
    print('==> doing epoch on training data:')
    print("==> online epoch # " .. epoch .. ' [batchSize = ' .. opt.batchSize .. ']') 


    for t = 1,trainData.Patches:size(1),opt.batchSize do

        -- disp progress
        xlua.progress(t, trainData.Patches:size(1))
        collectgarbage()

        -- batch fits?
        if (t + opt.batchSize - 1) > trainData.Patches:size(1) then
            break
        end

        -- create mini batch
        local idx = 1
        for i = t,t+opt.batchSize-1 do
            x1[idx] = trainData.Patches[shuffle1[i]]
            x2[idx] = trainData.Patches[shuffle2[i]]
            if trainData.ID[shuffle1[i]] == trainData.ID[shuffle2[i]] then
                similar[idx] = 1
            else
                similar[idx] = -1
            end

            idx = idx + 1
        end

        -- create closure to evaluate f(X) and df/dX
        local eval_E = function(w)
            -- reset gradients
            dE_dw:zero()

            -- evaluate function for complete mini batch
            local y = model:forward({x1,x2})
            --print(y1[1] - y2[2])
            -- estimate df/dW
            local D = loss:forward(y, similar)
            --print(numN)
            --print(numS)
            local dE_dy = loss:backward(y, similar)
            for p=1,opt.batchSize do
                if similar[p] == -1 then
                    DistanceN = DistanceN + y[p]--torch.dist(y1[p]:float(), y2[p]:float())
                    numN = numN +1
                else

                    numS = numS +1
                    DistanceS = DistanceS + y[p]--torch.dist(y1[p]:float(), y2[p]:float())
                end
            end
            model:backward({x1,x2},dE_dy)

            -- return f and df/dX
            return 0,dE_dw
        end

        -- optimize on current mini-batch
        optim.sgd(eval_E, w, optimState)



    end

    -- time taken
    time = sys.clock() - time
    time = time / trainData.Patches:size(1)
    print("\n==> time to learn 1 sample = " .. (time*1000) .. 'ms')

    numS = math.max(numS, 1)
    numN = math.max(numN, 1)
    DistanceS = DistanceS / numS
    DistanceN = DistanceN / numN

    print('Distance of similar patches = ' .. DistanceS)
    print('Distance of dissimilar patches = ' .. DistanceN)

    -- save/log current net
    local filename = paths.concat(opt.save, 'model.net')
    os.execute('mkdir -p ' .. sys.dirname(filename))
    print('==> saving model to '..filename)
    --torch.save(filename, w:float())
    torch.save(filename, model)

    -- next epoch
    epoch = epoch + 1
    return DistanceS, DistanceN
end

-- Export:
return train

