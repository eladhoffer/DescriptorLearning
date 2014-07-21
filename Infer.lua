require 'cunn'
require 'PyramidPacker'
require 'PyramidUnPacker'
require 'DistanceVector'
require 'image'
model = torch.load('./results/model.net')

scales = {1,0.7,0.5,0.3,0.2,0.1}

template = image.loadJPG('./template.jpg')
img = image.loadJPG('./image4.jpg')
img = (img-img:mean())/img:std()

template = image.scale(template,32,32):resize(1,3,32,32)
template = (template-template:mean())/template:std()

vector = model:forward(template:cuda())
model.modules[9] = nil
local Dist = nn.DistanceVector(vector,1)
packer = nn.PyramidPacker(model, scales)
unpacker = nn.PyramidUnPacker(model)


   pyramid, coordinates = packer:forward(img)

   multiscale = model:forward(pyramid:reshape(1,3,pyramid:size(2),pyramid:size(3)):cuda())
   multiscaleDist = Dist:forward(multiscale[1])
distanceMaps = unpacker:forward(multiscaleDist:resize(1,multiscaleDist:size(1),multiscaleDist:size(2)), coordinates)
