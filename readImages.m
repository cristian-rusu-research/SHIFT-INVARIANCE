function cI = readImages(images)
%% read images from a cell of paths

cI = [];
for i=1:length(images)
    disp(['processing ' char(images(i)) ' ...']);
    
    if (exist(char(images(i)), 'file') ~= 2)
        continue;
    end
    
    I = double(imread(char(images(i))));
    I = I(:, :, 1);
    [m n] = size(I);
    newn = n*m/64;

    C = mat2cell(I, 8*ones(1, m/8), 8*ones(1, n/8));
    [mc nc] = size(C);
    I = zeros(64, newn);
    
    index = 0;
    for indexi = 1:mc
        for indexj = 1:nc
            
            aux = reshape(C{indexi,indexj}, 1, 64);
            % remove mean?
%             aux = aux - mean(aux);

            % normalize?
%             aux = aux/norm(aux);

            % check extra condition
%             if (norm(aux)>0.1)
                index = index + 1;
                I(:,index) = aux;
%             end
        end
    end
    
    cI = [cI I(:, 1:index)];
end
