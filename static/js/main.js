$(function(){
  // 1) Autocompletar lat/lon al seleccionar Pozo
  $("#pozo").change(function(){
    let pozo = $(this).val();
    if(!pozo) return;
    $.getJSON("/coords/"+pozo, function(d){
      $("#lat").val(d.lat);
      $("#lon").val(d.lon);
    });
  });

  // 2) Generar inputs de cantidad por cada maniobra seleccionada
  $("#maniobras").change(function(){
    const selected = $(this).val() || [];
    const container = $("#cantidades");
    container.empty();

    // Para saber cuáles son del dataset2, exponelas en un array global en tu template:
    // <script>const maniobrasDataset2 = {{ maniobras_2|tojson }};</script>
    selected.forEach(function(m){
      if(window.maniobrasDataset2.includes(m)){
        // crea un input numérico para esta maniobra
        const id = "cant_" + m.replace(/\s+/g,"_");
        const field = `
          <div class="mb-2">
            <label for="${id}" class="form-label">${m}</label>
            <input type="number" step="0.01" name="pickup_weight_${m}" 
                   id="${id}" class="form-control" value="0">
          </div>`;
        container.append(field);
      }
    });
  });

  // 3) Antes de enviar, convertimos esos múltiples inputs
  $("#pred-form").submit(function(){
    const form = $(this);
    const selected = $("#maniobras").val() || [];

    // Añadimos un campo oculto por cada maniobra:
    // pickup_weight será un JSON serializado
    const weights = {};
    selected.forEach(function(m){
      const val = parseFloat($("[name='pickup_weight_"+m+"']").val()) || 0;
      weights[m] = val;
    });

    // quitamos cualquier input anterior llamado 'pickup_weights'
    form.find("input[name='pickup_weights']").remove();
    // lo añadimos
    form.append(
      $("<input>")
        .attr("type","hidden")
        .attr("name","pickup_weights")
        .val(JSON.stringify(weights))
    );
    return true;
  });
});

