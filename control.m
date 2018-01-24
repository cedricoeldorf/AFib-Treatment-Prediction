numfiles = 167;
mydata = cell(1, numfiles);
a = ones(2500,12*(numfiles));
count = 1;
for k = 1:numfiles
    
        myfilename = sprintf('patient_testing%dQRST_canc.mat', k);
        mydata{k} = importdata(myfilename);
        %k  don't forget to subtract 1 if necessary!
        a(:,count:(count+11)) = mydata{1,k};
        count = count+12;
    
end

count = 2;
sol = zeros(numfiles*3,1);
cc = 1; % counteerr
for i=1:numfiles
    bb  = 1;
    for j=1:3
        x = dominaanntt(a(bb:(bb+832),count)); % discarding 1st  seconnd
        bb = bb+833;
        sol(cc) = x;
        cc = cc+1;
    end
    count = count+12;
end
figure
plot(sol(:,1))
%plot(a(:,1),'--')