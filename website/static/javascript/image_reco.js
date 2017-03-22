function extract_uploaded_file(data, textStatus, jqXHR) {
	var file_name = data['file_name'];
	// $('#result').text(file_name);
	// console.log(data.res);
	$('#result').append('<img id="dynamic" height="300" width="300">');
	$('#dynamic').attr('src', flask_util.url_for("uploaded_file", {filename: file_name}));
	for (var i=0; i<data.res.length; i++){
		$('<p>',{
		text: data.res[i]}
		).appendTo('#result');
	}
}