function [ output] = convolution_layer(w,input)
inFeat = size(input,3);
inSize = size(input);
outFeat = size(w,4);

output = zeros(inSize(1) - size(w,1) +1, inSize(2) - size(w,2) +1, outFeat);
for ii=1:inFeat
    for io=1:outFeat
        output(:,:,io) = conv2(input(:,:,ii),w(:,:,ii,io),'valid');
    end
end

end

