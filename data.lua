require 'torch'
require 'nnx'
require 'image'

function Normalize(Data)
    local N = Data:size(1)
    for i=1,N do
        local Mean = Data[i]:mean()
        local Std = Data[i]:std()
        Data[i] = (Data[i] - Mean)/Std
    end
    return Data
end

--local train_data = torch.load('../svhn-original/train_32x32.t7', 'ascii')
--local test_data = torch.load('../svhn-original/test_32x32.t7', 'ascii')
--
--return {
--    trainData = {
--        Patches = Normalize(train_data.X:float()),
--        ID = train_data.y:squeeze()
--    },
--    testData = {
--        Patches = Normalize(test_data.X:float()),
--        ID = test_data.y:squeeze()
--    }
--}


rnd_rotate = function(angle_range)
	
        apply_rot = function(Data)
				angle = math.random(-angle_range,angle_range)
				mirrored = math.random(0,1)==1
				Data = image.rotate(Data,math.rad(angle))
				
      			if (mirrored) then
					Data = image.hflip(Data)
				end
				return Data
	end
	return apply_rot
end

function GenerateAugmentations(img, id, NumAugmentations)
    local augs = torch.Tensor(NumAugmentations, img:size(1), img:size(2), img:size(3))
    local ids = torch.Tensor(NumAugmentations):fill(id)
    for n=1,NumAugmentations do
        augs[n] = rnd_rotate(180)(img)
    end
    --augs[1] = img
    --augs[2] = image.hflip(img)
    --augs[3] = image.vflip(img)
    --augs[4] = image.hflip(augs[3])

    --Datalist.Patches
    return augs, id
end

-----------------------------------------CIFAR DATA-----------------------------------------------------
---- training/test size
print '==> loading dataset'
local BATCH_SIZE = 10000
local NUM_BATCHES = 1
local trainData = {
   data = torch.Tensor(NUM_BATCHES*BATCH_SIZE, 3, 32, 32),
   labels = torch.Tensor(NUM_BATCHES*BATCH_SIZE),
   size = function() return NUM_BATCHES*BATCH_SIZE end
}

for i=1,NUM_BATCHES do
    local train_file = '../../Datasets/cifar-10-batches-t7/data_batch_' .. tostring(i) .. '.t7'
    print('==> loading training file...' .. train_file);
    local loaded = torch.load(train_file, 'ascii')
    trainData.data:narrow(1,(i-1)*BATCH_SIZE+1,BATCH_SIZE):copy(loaded.data:transpose(1,2):reshape(BATCH_SIZE,3,32,32))
    trainData.labels:narrow(1,(i-1)*BATCH_SIZE+1,BATCH_SIZE):copy(loaded.labels[1] + 1)
end


local loaded = torch.load('../../Datasets/cifar-10-batches-t7/test_batch.t7','ascii')
local testData = {
   data = loaded.data:transpose(1,2):reshape(BATCH_SIZE,3,32,32),
   labels = loaded.labels[1] + 1,
   size = function() return BATCH_SIZE end
}

function BuildMatchList(NumImages,NumAugmentations, SizeList,ratio)

local MatchList = torch.Tensor(SizeList,2)
for i=1,math.ceil(SizeList/2) do
    local id = math.random(1,NumImages)
    local aug1 = math.random(1,NumAugmentations)
    local aug2 = math.random(1,NumAugmentations)
    MatchList[i][1] = (id-1)*NumAugmentations + aug1

    MatchList[i][2] = (id-1)*NumAugmentations + aug2
end
for i=math.ceil(SizeList/2)+1,SizeList do
    local id1 = math.random(1,NumImages)
    local id2 = math.random(1,NumImages)
    local aug1 = math.random(1,NumAugmentations)
    local aug2 = math.random(1,NumAugmentations)
    MatchList[i][1] = (id1-1)*NumAugmentations + aug1
    MatchList[i][2] = (id2-1)*NumAugmentations + aug2
end
return MatchList
end

----------------------------------------------------------------------
print '==> preprocessing data'

trainData.data = Normalize(trainData.data:float())

testData.data = Normalize(testData.data:float())
local NumAugmentations = 100
local NumImages = 1000
local TrainingPatches = torch.Tensor(NumAugmentations*NumImages, 3,32,32)
local TrainingID = torch.Tensor(NumImages*NumAugmentations)
for n=1,NumImages do
    TrainingPatches[{{(n-1)*NumAugmentations+1,n*NumAugmentations},{},{},{}}], TrainingID[{{(n-1)*NumAugmentations+1, n*NumAugmentations}}] = GenerateAugmentations(trainData.data[n], n, NumAugmentations)
end

local NumTestImages = 100
local TestPatches= torch.Tensor(NumAugmentations*NumTestImages, 3,32,32)
local TestID = torch.Tensor(NumTestImages*NumAugmentations)
for n=1,NumTestImages do
    TestPatches[{{(n-1)*NumAugmentations+1,n*NumAugmentations},{},{},{}}], TestID[{{(n-1)*NumAugmentations+1, n*NumAugmentations}}] = GenerateAugmentations(testData.data[n], n, NumAugmentations)
end

local NumTrainData = 50000
local NumTestData = 5000
local MatchListTraining = torch.Tensor(NumTrainData,2)
local MatchListTest = torch.Tensor(NumTestData,2)
return {
    trainData = {
        Patches = TrainingPatches:float(),
        ID = TrainingID:squeeze(),
        MatchList = BuildMatchList(NumImages, NumAugmentations, NumTrainData, 0.5)
    },
    testData = {
        Patches = TestPatches:float(),
        ID = TestID:squeeze(),
        MatchList = BuildMatchList(NumTestImages, NumAugmentations, NumTestData, 0.5)
    }
}


