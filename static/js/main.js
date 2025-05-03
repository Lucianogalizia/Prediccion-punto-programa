$(function(){
  // 1) Autocompletar lat/lon
  $("#pozo").change(function(){
    let pozo = $(this).val();
    if(!pozo) return;
    $.getJSON("/coords/"+pozo, function(d){
      $("#lat").val(d.lat);
      $("#lon").val(d.lon);
    });
  });

  // 2) Al cambiar selección de maniobras
  $("#maniobras").change(function(){
    const sel = $(this).val() || [];
    const cont = $("#cantidades");
    cont.empty();

    sel.forEach(function(m){
      if(maniobrasDataset2.includes(m)){
        // creamos un campo numérico para cada maniobra de Dataset2
        // name="pickup_weight_mi_maniobra"
        let safe = m.replace(/\s+/g,"_");
        cont.append(`
          <div class="mb-2">
            <label class="form-label">${m}</label>
            <input type="number" step="0.01" 
                   name="pickup_weight_${safe}" 
                   class="form-control" value="0">
          </div>
        `);
      }
    });
  });

  // 3) Al enviar, volcamos todas las cantidades en un JSON oculto
  $("#pred-form").submit(function(){
    const sel = $("#maniobras").val() || [];
    let weights = {};
    sel.forEach(function(m){
      let key = "pickup_weight_"+m.replace(/\s+/g,"_");
      let v = parseFloat($(`[name='${key}']`).val())||0;
      weights[m] = v;
    });
    // limpiamos cualquier input previo
    $(this).find("input[name='pickup_weights']").remove();
    // añadimos el JSON
    $(this).append(
      $("<input>")
        .attr("type","hidden")
        .attr("name","pickup_weights")
        .val(JSON.stringify(weights))
    );
    return true;
  });
});


