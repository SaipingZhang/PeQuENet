% YUV sequence -> RGB
function YUVtoRGB(filepath, str, row, col, frames)
fid = fopen(filepath,'r');
if exist(filepath(1:end-4))
   return
end
mkdir(filepath(1:end-4));
Y=zeros(row,col);
U=zeros(row/2,col/2);
V=zeros(row/2,col/2);
UU=zeros(row,col);
VV=zeros(row,col);


for frame=1:frames
    [Y(:,:),~] = fread(fid,[row,col],'uchar');
    [U(:,:),~]= fread(fid,[row/2,col/2],'uchar');
    [V(:,:),~]= fread(fid,[row/2,col/2],'uchar');
   
    
    UU(1:2:row-1,1:2:col-1)=U(:,:);
    UU(1:2:row-1,2:2:col)=U(:,:);
    UU(2:2:row,1:2:col-1)=U(:,:);
    UU(2:2:row,2:2:col)=U(:,:);
    
    
    VV(1:2:row-1,1:2:col-1)=V(:,:);
    VV(1:2:row-1,2:2:col)=V(:,:);
    VV(2:2:row,1:2:col-1)=V(:,:);
    VV(2:2:row,2:2:col)=V(:,:);
    
    
    R = Y + 1.140 * (VV-128 );
    G = Y + 0.395 * (UU-128 ) - 0.581 *(VV-128);
    B = Y + 2.032 *(UU-128);
    
    R(R(:,:)<0) = 0;
    R(R(:,:)>255) = 255;
    G(G(:,:)<0) = 0;
    G(G(:,:)>255) = 255;
    B(B(:,:)<0) = 0;
    B(B(:,:)>255) = 255;
    
    R=R/255;
    G=G/255;
    B=B/255;

    images(:,:,1)=R(:,:)';
    images(:,:,2)=G(:,:)';
    images(:,:,3)=B(:,:)';

    imwrite(images,[filepath(1:end-4),'\',str,'_',num2str(frame), '.png'])          

end
fclose(fid);