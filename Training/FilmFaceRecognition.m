(* ::Package:: *)

faceImageBase = {
   Import["https://ia.media-imdb.com/images/M/MV5BMjEzMjk4NDU4MF5BMl5BanBnXkFtZTcwMDMyNjQzMg@@._V1_UX214_CR0,0,214,317_AL_.jpg"]->"Daniel Craig",
   Import["http://ksassets.timeincuk.net/wp/uploads/sites/55/2017/10/GettyImages-689027928-920x584.jpg"]->"Eva Green"};


faceBase=Map[CZDLibFaceDescriptor[#[[1]]]->#[[2]]&,faceImageBase];
