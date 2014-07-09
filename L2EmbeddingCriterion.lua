local L2EmbeddingCriterion, parent = torch.class('nn.L2EmbeddingCriterion', 'nn.Criterion')


--[[Attention!!! updateoutput must be invoked before gradinput]]
function L2EmbeddingCriterion:__init(M, avg)
    parent.__init(self)
    self.output = torch.Tensor()
    self.sizeAverage = avg or false
    self.marginDist = M or 10
    self.Maximum = maximum or 999999
end

function L2EmbeddingCriterion:updateOutput(input, target)
    self.output = torch.add(input, torch.mul(target,-1))
    self.output = self.output:cmul(self.output):sum(2)
    
    return self.output:squeeze()
end

function L2EmbeddingCriterion:updateGradInput(input, target, similar)
    local similar = similar-- or torch.ones(input:size(1),1)
    self.gradInput = torch.add(input, torch.mul(target,-1)):mul(2)
    --if similar:nDimension() == 1 then
    --    similar = similar:resize(similar:size(1),1)
    --end
    --local idx_diff = similar:le(0):expandAs(input)
    --local M = self.marginDist
    --self.gradInput[idx_diff] = self.gradInput[idx_diff]:mul(-1)
    --local idx_small = idx_diff:cmul(torch.abs(self.output):expandAs(input):lt(M))

    --self.gradInput[idx_small] = torch.sign(self.gradInput[idx_small]):mul(M)

    return self.gradInput
end


function L2EmbeddingCriterion:backward(input, target, similar)
   return self:updateGradInput(input, target, similar)
end
