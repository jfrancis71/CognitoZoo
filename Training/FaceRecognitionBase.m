(* ::Package:: *)

faceImageBase={Import["http://www.irishnews.com/picturesarchive/irishnews/irishnews/2017/06/23/103007301-d527c5e7-c4d9-4207-bfbd-358b5eb82476.jpg"]->"David Dimbleby",
   Import["https://pbs.twimg.com/media/DY-UpQoW0AI8xRr.jpg"]->"Paris Lees",
   Import["https://lifeondoverbeach.files.wordpress.com/2015/10/peter-hitchens.jpg"]->"Peter Hitchins",
   Import["https://s3-eu-central-1.amazonaws.com/centaur-wp/l2bthelawyer/prod/content/uploads/2016/03/08163100/Screen-Shot-2016-06-01-at-09.51.59.png"]->"Shami Chakrabarti",
   Import["https://cdn.images.express.co.uk/img/dynamic/20/590x/secondary/GMB-barry-gardiner-932002.jpg"]->"Barry Gardiner",
   Import["https://secure.i.telegraph.co.uk/multimedia/archive/02535/mar_2535779b.jpg"]->"Andrew Marr"
};


faceBase=Map[CZDLibFaceDescriptor[#[[1]]]->#[[2]]&,faceImageBase];


Export["c:\\Users\\julian\\Google Drive\\Personal\\Computer Science\\CZModels\\FaceBase.mx",faceBase];
