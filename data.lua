require 'torch'

function Normalize(Data)
    local N = Data:size(1)
    for i=1,N do
        local Mean = Data[i]:mean()
        local Std = Data[i]:std()
        Data[i] = (Data[i] - Mean)/Std
    end
    return Data
end

local train_data = torch.load('../svhn-original/train_32x32.t7', 'ascii')
local test_data = torch.load('../svhn-original/test_32x32.t7', 'ascii')

return {
    trainData = {
        Patches = Normalize(train_data.X:float()),
        ID = train_data.y:squeeze()
    },
    testData = {
        Patches = Normalize(test_data.X:float()),
        ID = test_data.y:squeeze()
    }
}
