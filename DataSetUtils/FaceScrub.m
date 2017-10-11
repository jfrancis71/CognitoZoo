(* ::Package:: *)

(* Suggested use:
files=FileNames["C:\\Users\\julian\\ImageDataSets\\FaceScrub\\ActorImages\\ActorImages1\\*.jpg"];Length[files]
p1 = Partition[files,UpTo[100]] ;
images = Flatten[CZGetImagesPartition[ p1 ]];
faceScrub = CZImportFaceScrub[ files, images, CZReadFaceScrubDatabase[ "C:\\Users\\julian\\ImageDataSets\\FaceScrub\\facescrub_actors.txt"] ];
*)


CZReadFaceScrubDatabase[srcFile_] := ReadList[StringToStream[Import[srcFile]],String,RecordSeparators->"\n"];


CZGetURL[recordNo_Integer,database_] := (
   fileName=ReadList[StringToStream[database[[recordNo]]],Record,RecordSeparators->"\t"][[4]];
   Import[fileName]
)


CZGetName[recordNo_Integer,database_] := ReadList[StringToStream[database[[recordNo]]],Record,RecordSeparators->"\t"][[1]]


CZBoundingBox[recordNo_Integer,database_,image_] := ( 
   c=ReadList[StringToStream[database[[recordNo]]],Record,RecordSeparators->"\t"][[5]];
   p=Map[FromDigits,ReadList[StringToStream[c],Record,RecordSeparators->","]];
   {{p[[1]],ImageDimensions[image][[2]]-p[[4]]},{p[[3]],ImageDimensions[image][[2]]-p[[2]]}})


CZImportFaceScrub[file_, database_] := {Import[file],CZGetName[FromDigits[FileBaseName[file]],database],CZBoundingBox[FromDigits[FileBaseName[file]],database,Import[file]]}


(* Implemented as Table as opposed to Map. Easier to track progress *)
CZImportFaceScrub[files_,images_,database_] := (
   Assert[ Length[files] == Length[images] ];
   Table[{images[[f]],CZGetName[FromDigits[FileBaseName[files[[f]]]],database],CZBoundingBox[FromDigits[FileBaseName[files[[f]]]],database,images[[f]]]},{f,1,Length[files]}]
)


(* Internal function to deal with speed issues on loading many images *)
CZGetImagesPartition[setOfFiles_] := Table[Map[Import,setOfFiles[[f]]],{f,1,Length[setOfFiles]}];


CZGetImages[files_] := CZGetImagesPartition[ Partition[files,UpTo[100]] ]


CZGetRecordNo[file_] := FromDigits[FileBaseName[file]]
