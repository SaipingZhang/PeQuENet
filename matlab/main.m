path = 'D:\YUVSequences\';
files = dir(path);
for i = 1 : length(files)
    if files(i).isdir == 0
        str = files(i).name;
        filepath = fullfile(files(i).folder, files(i).name);
        id1 = findstr(files(i).name,'_');
        id2 = findstr(files(i).name,'x');
        id3 = findstr(files(i).name,'.');
        row = str2num(str(id1(1)+1:id2-1));
        col = str2num(str(id2+1:id1(2)-1));
        frame = str2num(str(id1(2)+1:id3-1));
        fprintf("%s, %d, %d, %d\n", filepath, row, col, frame);
        YUVtoRGB(filepath, str, row, col, frame);
    end
end