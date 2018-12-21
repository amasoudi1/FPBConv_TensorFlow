clear 
close 

%% preprocess the images
load('bags400_50_test_rescaled.mat')

for i = 1:400
    ti = f(:,i);    
    ti = imresize(reshape(ti,50,50),[64 64]);
    ti(ti < 0) = 0;
%     figure(1)
%     imagesc(f)
%     colormap(gray)
%     axis image
%     axis off
    txt = [num2str(i) '_gt.mat'];
    cd('./test/grount')
    save(txt,'ti')
    cd ..
    cd ..
    theta = (0:10:179);
    R = radon(ti,theta);
    fi = iradon(R,theta,64);
    fi(fi < 0) = 0;
    
    figure(2)
    imagesc(fi)
    colormap(gray)
    axis image
    axis off
    txt = [num2str(i) '_r.mat'];
    cd('./test/recon')
    save(txt,'fi')
    cd ..
    cd ..
end