
% currentFolder = uigetdir('D:\tooldata','Select a CSV folder to load');
% DATFile = strcat(currentFolder, '\*.xlsx');
%     files=dir(DATFile); 
    
    clear all;
    close all;
%%    
%     num=zeros(1,144);  % 144 for 2048T
   
%     filename='stat001_simp.las';
 
  data4train=zeros(1,4);
   files=dir('*Sta006*.las');
%     for k=1:length(files)
         num=zeros(1,197);  % 52 for 128T
%      filename = files(k).name;
          filename = files.name;
%     num1=LoadDumpDataCSV_V2(filename);
    [num1,curve_info] = loadlasdata(filename);
    num=vertcat(num,num1);
%     end
    
%     files=dir('*ori*.las');
%      k=1;
%      filename = files(k).name;
%    [channelData,curve_info] = loadlasdata(filename);
   channelName=curve_info(:,1);
%     files='stat1.xlsx';
%     num1=xlsread(files,'Sheet1');
%     files=dir('*simp*.las');;
%    for k=1:length(files)
%     filename = files(k).name;
%     num1=xlsread(filename,'128T');
%     num1=num1(4:end,:);
%      num=vertcat(num,num1);
%    end
   
% %     FolderName = currentFolder;
% %     FullInputFileName = strcat(FolderName,'\',inputFileName);
% %     
% %     fileID = fopen(FullInputFileName,'r');
% save('channellist.mat','channelName');
channelData=num;

% EnvChannel={'LSVMTRCURMS1','LSVMTRDCMS1','LSVSTATUSMS1','LSVPOSMS1','LSVACTVLTMS1','LSVTEMPMS1'}';
EnvChannel={'LSVMTRCURMS1','LSVMTRDCMS1','LSVPOSMS1','LSVSTATUSMS1'}';


% EnvChannel={'CV1MTRCURMS1','CV2MTRCURMS1','CV3MTRCURMS1','CV4MTRCURMS1','CV5MTRCURMS1','CV6MTRCURMS1','LSVMTRCURMS1', ...
%     'USVMTRCURMS1','CV1POSMS1','CV2POSMS1','CV3POSMS1','CV4POSMS1','CV5POSMS1','CV6POSMS1','LSVPOSMS1','USVPOSMS1'}';
        EnvChannelData = [];
        EnvChannelName = {};
        for i=1:length(EnvChannel)
            EnvChannelIdx= find(strcmp(channelName,EnvChannel{i})==1);
            if ~isempty(EnvChannelIdx)
%               EnvChannelData =[EnvChannelData cell2double(channelData{:,EnvChannelIdx})];
            EnvChannelData =[EnvChannelData channelData(:,EnvChannelIdx(1))];
            EnvChannelName =[EnvChannelName channelName(EnvChannelIdx(1))];
            end
        end
        
        num = EnvChannelData;
    
 %  
   indx1=unique([find(num(:,1)==-999.2500) find((num(:,1)==65535))  find(isnan(num(:,1)))]);
            indx2=unique([find(num(:,2)==-999.2500) find((num(:,2)==65535))  find(isnan(num(:,2)))]);
             indx3=unique([find(num(:,3)==-999.2500) find((num(:,3)==65535))  find(isnan(num(:,3)))]);
              indx4=unique([find(num(:,4)==-999.2500) find((num(:,4)==65535))  find(isnan(num(:,4)))]);
%             indx5=unique([find(num(:,5)==-999.2500) find((num(:,5)==65535))  find(isnan(num(:,5)))]);
%             indx6=unique([find(num(:,6)==-999.2500) find((num(:,6)==65535))  find(isnan(num(:,6)))]);
%             indx7=unique([find(num(:,7)==-999.2500) find((num(:,7)==65535))  find(isnan(num(:,7)))]);
%             indx8=unique([find(num(:,8)==-999.2500) find((num(:,8)==65535))  find(isnan(num(:,8)))]);
%             indx9=unique([find(num(:,9)==-999.2500) find((num(:,9)==65535))  find(isnan(num(:,9)))]);
%             indx10=unique([find(num(:,10)==-999.2500) find((num(:,10)==65535))  find(isnan(num(:,10)))]);
%             indx11=unique([find(num(:,11)==-999.2500) find((num(:,11)==65535))  find(isnan(num(:,11)))]);
%             indx12=unique([find(num(:,12)==-999.2500) find((num(:,12)==65535))  find(isnan(num(:,12)))]);
%             indx13=unique([find(num(:,13)==-999.2500) find((num(:,13)==65535))  find(isnan(num(:,13)))]);
%             indx14=unique([find(num(:,14)==-999.2500) find((num(:,14)==65535))  find(isnan(num(:,14)))]);
%             indx15=unique([find(num(:,15)==-999.2500) find((num(:,15)==65535))  find(isnan(num(:,15)))]);
%             indx16=unique([find(num(:,16)==-999.2500) find((num(:,16)==65535))  find(isnan(num(:,16)))]);
%              x=union(union(union(union(union(indx1,indx2),indx3),indx4),indx5),indx6);
%              y=union(union(union(union(union(indx7,indx8),indx9),indx10),indx11),indx12);
%              index3=union(x,y);
               index3=union(union(union(indx1,indx2),indx3),indx4);
%              index3=union(union(union(union(union(indx1,indx2),indx3),indx4),indx5),indx6);
%             index3=union(union(union(union(indx1,indx2),indx3),indx5),indx6);
             num(index3,:)=[];
 % num(2:10)=1e7 is a spike remove 
 num(1:2,:)=[];
 
data4train=vertcat(data4train,num);
 %% in order on how I built this ctf1 run1 train data
  all4train=zeros(1,4);
%  all4train=vertcat(data4train(260:300,:),all4train);   % from sta001

% all4train=vertcat(data4train(30730:30765,:),all4train);  % from sta002
% all4train=vertcat(data4train(5365:5390,:),all4train);  % from sta002


% all4train=vertcat(data4train(33340:33380,:),all4train);  % from sta004
% all4train=vertcat(data4train(26360:26400,:),all4train);  % from sta004
% all4train=vertcat(data4train(19820:19860,:),all4train);  % from sta004
% all4train=vertcat(data4train(9800:9860,:),all4train);  % from sta004

all4train=vertcat(data4train(35840:35900,:),all4train);  % from sta006
all4train=vertcat(data4train(34840:34900,:),all4train);  % from sta006
all4train=vertcat(data4train(34610:34660,:),all4train);  % from sta006
all4train=vertcat(data4train(34420:34460,:),all4train);  % from sta006
all4train=vertcat(data4train(32460:32520,:),all4train);  % from sta006
all4train=vertcat(data4train(29420:29470,:),all4train);  % from sta006
all4train=vertcat(data4train(28560:28640,:),all4train);  % from sta006
all4train=vertcat(data4train(28470:28520,:),all4train);  % from sta006
all4train=vertcat(data4train(28000:28080,:),all4train);  % from sta006
all4train=vertcat(data4train(27080:27140,:),all4train);  % from sta006
all4train=vertcat(data4train(24200:24250,:),all4train);  % from sta006
all4train=vertcat(data4train(20760:20820,:),all4train);  % from sta006
all4train=vertcat(data4train(17400:17440,:),all4train);  % from sta006
all4train=vertcat(data4train(2420:2450,:),all4train);  % from sta006
all4train=vertcat(data4train(230:260,:),all4train);  % from sta006

 all4train(end,3)=100;

%%
%       data4train1=zeros(1,4);
 data4train1=vertcat(data4train1,all4train);

%%
%   save('ctf1run1lsv4train.mat','all4train')
% data4train=vertcat(data4train,num);
% end

 %%
% %  %  data4train1(255:295,:)=[];  % logic was not correct after conscating
% %  %  them, so have to remove the ilogical ones
% % % data4train1(1:45,:)=[];
% %  all4train=data4train1;
% %   save('ctf1run1lsv4trainJan03.mat','all4train')
% %   %%
% %    load ctf1run1usv4trainJan03.mat;
% %   data4train1=vertcat(data4train1,all4train);
% %  %% % now combine lsv and usv for ctf1run1, remove the illogical ones
% %   data4train1(1340:1460,:)=[];
% %   data4train1(1020:1070,:)=[];
% %   data4train1(1:30,:)=[];
% %  %% save to lsvusv train data from ctf1run1
% %  all4train=data4train1;
% %   save('ctf1run1usvlsv4trainJan03.mat','all4train')
 %%
% load ctf1run1usvlsv4trainJan03.mat;
%%

%%
load ctf1run1usv4trainJan03.mat;
data4train2=all4train(:,:);
% data4train2=data4train1(:,:);
cur4train=data4train2(:,1:2)';
pos4train=data4train2(:,3)';
stattrain=data4train2(:,4)';
% actvtrain=data4train1(:,5)';
% temptrain=data4train1(:,6)';

% x=num2cell(cur4train);
% y=num2cell(pos4train);
%


figure  % to plot 6 signals
subplot(4,1,1)
plot(cur4train(1,:),'r')
title('current');
subplot(4,1,2)
plot(cur4train(2,:),'b')
title('Duty cycle');
subplot(4,1,3)
plot(pos4train,'g')
title('valve position')
subplot(4,1,4)
plot(stattrain,'c')
title('status word')
% subplot(6,1,5)
% plot(actvtrain,'k')
% title('valve ACT voltage')
% subplot(6,1,6)
% plot(temptrain,'m')
% title('temperature')
% title('CTF run1 R01 Sta002')
% figure
% subplot(2,1,1)
% plot(cur4train,'r')
% title('input');
% subplot(2,1,2)
% plot(pos4train,'g')
% title('output')

% %%
% %%
%