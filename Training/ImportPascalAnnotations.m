(* ::Package:: *)

(*
   Reads in a Pascal VOC annotation string and returns a list of objects in format { {rect1,obj1}, {rect2,obj2},...}
   Don't forget on import to import with option "XMLElement"
*)
CZImportPascalAnnotations[ fileName_String ] := CZReadPascalAnnotations@Import[fileName, "XMLElement" ];
CZReadPascalAnnotations[xml_] := Map[
   CZReadVOCXMLObject[#,FromDigits[Cases[Cases[xml[[1,3]],XMLElement["size",_,_]][[1,3]],XMLElement["height",_,_]][[1,3,1]]]]&,
   Cases[xml[[1,3]],XMLElement["object",_,_]][[All,3]]]


(* Bounding boxes are stored as y=0 means top of image, so need to convert *)
CZReadVOCXMLObject[xml_,height_] := {
   Rectangle[{FromDigits[Cases[First[Cases[xml,XMLElement["bndbox",_,_]]][[3]],XMLElement["xmin",_,_]][[1,3]]//First],
   height-FromDigits[Cases[First[Cases[xml,XMLElement["bndbox",_,_]]][[3]],XMLElement["ymax",_,_]][[1,3]]//First
   ]
   },
   {FromDigits[Cases[First[Cases[xml,XMLElement["bndbox",_,_]]][[3]],XMLElement["xmax",_,_]][[1,3]]//First],
   height-FromDigits[Cases[First[Cases[xml,XMLElement["bndbox",_,_]]][[3]],XMLElement["ymin",_,_]][[1,3]]//First
   ]
   }],
   Cases[xml,XMLElement["name",_,_]][[1,3,1]]
}
