// define jQuery
jQuery(function($){

var example = {

	elms: {
	},

	example: function() {
	},

	init: function() {
	}
};

//  ***********************
//  $$ Document ready
//  ***********************
$(function() {
	window.windowWidth = window.innerWidth;
	window.windowHeight = window.innerHeight;

	window.isiPhone = navigator.userAgent.toLowerCase().indexOf('iphone');
	window.isiPad = navigator.userAgent.toLowerCase().indexOf('ipad');
	window.isiPod = navigator.userAgent.toLowerCase().indexOf('ipod');

	window.tablet = 640;
	window.tabletWide = 800;
	window.desktop = 1024;

	// Add functions here

	// example.js
	example.init();

	$('#nav-icon').click(function(){
		$(this).toggleClass('open');
	});

});
//  ***********************
//  $$ Smart resize
//  ***********************
(function ($, sr) {
	var debounce = function (func, threshold, execAsap) {
		var timeout;
		return function debounced() {
			var obj = this,
				args = arguments;
			function delayed() {
				if (!execAsap) {
					func.apply(obj, args);
				}
				timeout = null;
			}
			if (timeout) {
				clearTimeout(timeout);
			} else if (execAsap) {
				func.apply(obj, args);
			}
			timeout = setTimeout(delayed, threshold || 500);
		};
	};
	jQuery.fn[sr] = function (fn) {
		return fn ? this.bind('resize', debounce(fn)) : this.trigger(sr);
	};
})(jQuery, 'smartresize'); // End smartresize


$(window).smartresize(function () {
	window.newWindowWidth = window.innerWidth;
	window.newWindowHeight = window.innerHeight;

	//	responsive-helpers.js


	window.windowWidth = window.innerWidth;
	window.windowHeight = window.innerHeight;
});
});// jQuery end
// no writing in this file