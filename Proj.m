function [out] = Proj(in,verticies)
if(length(in) ~= length(verticies))
   error('Something went wrong check the dimension of input vector with the verticies')
end 
for i=1:length(in)
mini=verticies{i}(1);
maxi=verticies{i}(2);
out(i)=min(maxi, max(mini, in(i)));
end
out=reshape(out,size(in));
end

