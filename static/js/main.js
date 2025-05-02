$(function(){
  $("#pozo").change(function(){
    let p = $(this).val();
    if(!p) return;
    $.getJSON("/coords/"+p, d=>{
      $("#lat").val(d.lat); $("#lon").val(d.lon);
    });
  });
  $("select[name='MANIOBRAS_NORMALIZADAS']").change(function(){
    let mani = $(this).val();
    // asume que maniobras_2 está expuesto en JS si hace falta;
    // o bien habilitamos siempre y validamos backend.
    if(maniosDataset2.includes(mani)){
      $("#pickup").prop("disabled",false);
    } else {
      $("#pickup").prop("disabled",true).val(0);
    }
  });
});
