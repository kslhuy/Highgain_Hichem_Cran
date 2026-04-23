function stateNext = stateModel(state,dt)
    F = [1 dt 0  0; 
         0  1 0  0;
         0  0 1 dt;
         0  0 0  1];
    stateNext = F*state;
end