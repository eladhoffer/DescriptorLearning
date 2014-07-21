local BatchHingeEmbeddingCriterion, parent = torch.class('nn.BatchHingeEmbeddingCriterion', 'nn.Criterion')

function BatchHingeEmbeddingCriterion:__init(margin)
   parent.__init(self)
   margin=margin or 1 
   self.margin = margin 
   self.gradInput = torch.Tensor()
   self.output = torch.Tensor()
end 
 
function BatchHingeEmbeddingCriterion:updateOutput(input,y)

    self.output = input:clone()
   for l=1,y:size(1) do
   
   if y[l]==-1 then
	 self.output[l] = math.max(0,self.margin - self.output[l])
 end
   end
   return self.output
end

function BatchHingeEmbeddingCriterion:updateGradInput(input, y)
  self.gradInput=y
  local dist = input
  for l=1,y:size(1) do
  if y[l] == -1 and  dist[l] > self.margin then
     self.gradInput[l]=0;
  end
  return self.gradInput 
end
end
