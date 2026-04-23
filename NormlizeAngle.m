function result = NormlizeAngle(angle0)
result=[];
angle=mod(angle0,2*pi);
for i=1:length(angle)
if(angle(i) > pi)
    result =[result angle(i) - 2 * pi];
else
    result =[result angle(i)];
end
end
