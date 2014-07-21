
local DistanceVector, parent = torch.class('nn.DistanceVector', 'nn.Module')
require 'BatchPairwiseDistance'
function DistanceVector:__init(vector,p)
   parent.__init(self)

   -- state
   self.output = torch.Tensor()
   self.vector = vector
   self.diff = torch.Tensor()
   self.norm=p
end 
  
function DistanceVector:updateOutput(input)
    local original_size = input:size()
   if input:dim() == 3 then
       local m = nn.BatchPairwiseDistance(self.norm):cuda()
       input = input:resize(input:size(1),input:size(2)*input:size(3)):t()
       local vector_expanded = self.vector:expand(original_size[2]*original_size[3], self.vector:size(2))
      self.output = m:forward({input, vector_expanded})
      self.output:resize(original_size[2], original_size[3])

      input = input:t():resize(original_size[1], original_size[2], original_size[3])
      
  else
      error('input must be vector or matrix')
   end  
 
   return self.output
end
-- save away Module:type(type) for later use.
DistanceVector._parent_type = parent.type

-- Fix the bug where tmp = nn.DistanceVector:cuda() fails to convert table
-- contents.  We could, and probably should, change Module.lua to loop over
-- and convert all the table elements in a module, but that might have 
-- repercussions, so this is a safer solution.
function DistanceVector:type(type)
   self:_parent_type(type)  -- Call the parent (Module) type function
   -- Now convert the left over table elements
   return self
end

