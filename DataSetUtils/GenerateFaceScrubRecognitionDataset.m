(* ::Package:: *)

<<DataSetUtils\FaceScrub.m


<<CZFaceDetection.m


database=CZReadFaceScrubDatabase[ "C:\\Users\\julian\\ImageDataSets\\FaceScrub\\facescrub_actresses.txt"];


files=FileNames["C:\\Users\\julian\\ImageDataSets\\FaceScrub\\ActressImages\\ActressImages6\\*.jpg"];Length[files]


images=Import["C:\\Users\\julian\\ImageDataSets\\FaceScrub\\ActressImages\\ActressImages6\\*.jpg"];Length[images]


faces=Table[FindFaces[images[[f]]],{f,1,Length[images]}];


getFace[faces_,rect_]:=Last@SortBy[faces,CZIntersectionOverUnion[#,rect]&]


getFaceImage[image_,faces_,rect_]:=ImageResize[ImageTrim[image,getFace[faces,rect]],64]


training1=Table[
If[Length[faces[[f]]]==0,{},
getFaceImage[images[[f]],faces[[f]],CZBoundingBox[CZGetRecordNo[files[[f]]],database,images[[f]]]]->CZGetName[CZGetRecordNo[files[[f]]],database]],{f,1,Length[faces]}];


training2=Delete[training1,Position[training1,{}]];


proc=Map[CZFaceNet[{ImageData[ColorConvert[ImageResize[#,{32,32}],"GrayScale"]]}][[1,1,1]]&,training2[[All,1]]];//AbsoluteTiming


training3=Delete[training2,Position[proc,x_/;x<.5]];


Export["c:\\Users\\julian\\ImageDataSets\\FaceScrub\\Actress6.mx",training3]


(* ::Input:: *)
(*m1=Import["c:\\Users\\julian\\ImageDataSets\\FaceScrub\\Actor1.mx"];*)
(*m2=Import["c:\\Users\\julian\\ImageDataSets\\FaceScrub\\Actor2.mx"];*)
(*m3=Import["c:\\Users\\julian\\ImageDataSets\\FaceScrub\\Actor3.mx"];*)
(*m4=Import["c:\\Users\\julian\\ImageDataSets\\FaceScrub\\Actor4.mx"];*)
(*m5=Import["c:\\Users\\julian\\ImageDataSets\\FaceScrub\\Actor5.mx"];*)
(*m6=Import["c:\\Users\\julian\\ImageDataSets\\FaceScrub\\Actor6.mx"];*)


(* ::Input:: *)
(*f1=Import["c:\\Users\\julian\\ImageDataSets\\FaceScrub\\Actress1.mx"];*)
(*f2=Import["c:\\Users\\julian\\ImageDataSets\\FaceScrub\\Actress2.mx"];*)
(*f3=Import["c:\\Users\\julian\\ImageDataSets\\FaceScrub\\Actress3.mx"];*)
(*f4=Import["c:\\Users\\julian\\ImageDataSets\\FaceScrub\\Actress4.mx"];*)
(*f5=Import["c:\\Users\\julian\\ImageDataSets\\FaceScrub\\Actress5.mx"];*)
(*f6=Import["c:\\Users\\julian\\ImageDataSets\\FaceScrub\\Actress6.mx"];*)


(* ::Input:: *)
(*j=Join[m1,m2,m3,m4,m5,m6,f1,f2,f3,f4,f5,f6];*)


g=GatherBy[j[[1;;-1]],#[[2]]&];//AbsoluteTiming


g[[4]];


Dynamic[l]


(* ::Input:: *)
(*t=Table[Union[g[[l]],SameTest->(ImageDistance[ColorConvert[#1[[1]],"Grayscale"],ColorConvert[#2[[1]],"Grayscale"]]<1&)],{l,1,Length[g]}]//Flatten;//AbsoluteTiming*)


Export["c:\\Users\\julian\\ImageDataSets\\FaceScrub\\FaceIdentityTraining.mx",t];
