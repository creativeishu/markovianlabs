function extract_uploaded_file(data, textStatus, jqXHR) {
	var file_name = data['file_name'];
	// $('#result').text(file_name);
	// console.log(data.res);
	$('#result').empty();
	$('#result').append('<div id="inputImage"></div>');
	$('#inputImage').append('<img id="dynamic" height="500" width="500">');
	$('#dynamic').attr('src', flask_util.url_for("uploaded_file", {filename: file_name}));
	$('#result').append('<div id="resultTable"></div>');
	for (var i=0; i<data.res.length; i++){
		$('<p>',{
		text: data.res[i]}
		).appendTo('#resultTable');
	}
}