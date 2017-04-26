// define jQuery
jQuery(function($){

var validation = {

	setMethods: function() {
		// Define error message options (format and position relative to error input)
		$.validator.setDefaults({
			debug: true,
			success: 'valid',
			errorElement: 'span',
			errorPlacement: function(error, element) {
				if ( $(element).is(':checkbox') ) { // Move message on checkboxes
					error.insertAfter( $(element).parent() );
				} else if ($(element).is(':radio')) { // Move message on radio buttons
					error.insertAfter( $(element).parent().parent() );
				} else if ($(element).siblings('.tooltip-wrapper').length) { // Move message when tooltip on the side
					error.insertAfter( $(element).parent() );
				} else if ($(element).attr('data-validation-nomessage')) { // Remove validation message
					error.remove();
				} else if ($(element).is('.home-panel-form-text')) { // home page registration panel
					error.insertAfter( $(element).siblings('.button') );
				} else { // This is the default behavior of the script
					error.insertAfter( $(element) );
				}
			}
		});

		// Validation method for regex expressions
		$.validator.addMethod(
			'regexFormat',
			function(value, element) {
				var regexString = $(element).attr('data-validation-regex');
				var check = false;
				var re = new RegExp(regexString);
				return this.optional(element) || (re.test(value) > 0);
			}
		);

		// Validation method for inputs that need matching other inputs
		$.validator.addMethod(
			'matchInput',
			function(value, element){
				var $this = $(element);
				var $thisVal = $this.val();
				var $matchEl = $this.attr('data-validation-matchTo');
			    var $matchElVal = $this.closest('form').find('[name="' + $matchEl + '"]').val();

			    if ($thisVal !== $matchElVal) {
			        return false;
			    } else {
			        return true;
			    }
			}
		);

		// Validation method for custom dropdowns
		$.validator.addMethod(
			'selectRequired',
			function(value, element){
				if (value === 'clearSelect' || value === '') {
					$(element).closest('.select-wrapper').find('.dk-selected').addClass('select-error');
					return false;
				} else {
					$(element).closest('.select-wrapper').find('.dk-selected').removeClass('select-error');
					return true;
				}
			}
		);

		// Validation method for min length
		$.validator.addMethod(
			'minLengthCheck',
			function(value, element) {
				var $lengthVal = $(element).attr('data-validation-minLength');
				var $inputValLength = value.length;
				if ($inputValLength < parseInt($lengthVal)) {
					return false;
				} else {
					return true;
				}
			}
		);

		// Validation method for max length
		$.validator.addMethod(
			'maxLengthCheck',
			function(value, element) {
				var $lengthVal = $(element).attr('data-validation-maxLength');
				var $inputValLength = value.length;
				if ($inputValLength > parseInt($lengthVal)) {
					return false;
				} else {
					return true;
				}
			}
		);

		// Validation method for date of birth
		$.validator.addMethod(
			'validBirthDate',
			function(value, element){
				var date = value;
				var dateSplit = date.split('/');

				function isValidDate2(y, m, d) {
					// No leap year by default
					var daysInMonth = [31,28,31,30,31,30,31,31,30,31,30,31];

					// Check for leap (if evenly divisible by 4)
					if (((y % 4 === 0) && (y % 100 !== 0)) || (y % 400 === 0)) {
						daysInMonth[1] = 29;
					}
					return d <= daysInMonth[--m];
				}

				if (
					(isValidDate2(dateSplit[0], dateSplit[1], dateSplit[2]) === false) || (dateSplit[0] === '' || dateSplit[1] === '' || dateSplit[2] === '')
				) {
					$('.birthday-selects').find('.dk-selected').each(function() {
						$(this).addClass('select-error');
					});
					$('.birthday-selects').find('select').each(function() {
						$(this).addClass('error');
					});
					return false;
				} else {
					$('.birthday-selects').find('.dk-selected').each(function() {
						$(this).removeClass('select-error');
					});
					$('.birthday-selects').find('select').each(function() {
						$(this).removeClass('error');
					});
					return true;
				}
			}
		);


		// Allow empty fields
		$.validator.addMethod(
			'allowEmpty',
			function(value, element) {
				var $inputValLength = value.length;
				if ($inputValLength === 0) {
					return true;
				}
			}
		);
	},

	// Trigger validation for each form
	validateForms: function() {

		$('form').each(function(){
			var $form = $(this);
			$form.validate({
				submitHandler: function(form) {
					$form.find('input[type="submit"]').prop('disabled', true);
					form.submit();
				}
			});
		});
	},

	// Set validation rules for different types input sets
	setRules: function() {
		// Check required
		$('input').each(function() {
			var message_required = $(this).attr('data-validation-empty');
			if (message_required) {
			    $(this).rules('add', {
					required: true,
			        messages: { required: message_required }
			    });
			}
		});

		// Regex input formats
		$('[data-validation-regex]').each(function() {
			var message_format = $(this).attr('data-validation-format');
		    $(this).rules('add', {
				regexFormat: true,
		        messages: {
		        	regexFormat: message_format
		        }
		    });
		});

		// Re-enter inputs matching (email and password)
		$('[data-validation-matchTo]').each(function() {
			var matchTo = $(this).attr('data-validation-matchTo');
			var message_matchTo = $(this).attr('data-validation-format');
		    $(this).rules('add', {
				matchInput: true,
				messages: {
					matchInput: message_matchTo
		        }
		    });
		});

		// Minimum length inputs
		$('[data-validation-minLength]').each(function() {
			var minLength_message = $(this).attr('data-validation-minLenght-message');
		    $(this).rules('add', {
				minLengthCheck: true,
				messages: {
					minLengthCheck: minLength_message
		        }
		    });
		});

		// Maximum length inputs
		$('[data-validation-maxLength]').each(function() {
			var maxLength_message = $(this).attr('data-validation-maxLenght-message');
		    $(this).rules('add', {
				maxLengthCheck: true,
				messages: {
					maxLengthCheck: maxLength_message
		        }
		    });
		});

		// Dropdowns
		$('select[data-validation-empty]').each(function() {
			var message_required = $(this).attr('data-validation-empty');
		    $(this).rules('add', {
				selectRequired: true,
				messages: {
					selectRequired: message_required
		        }
		    });
		});

		// Validate Date Of Birth
	    $('.dob-hidden').each(function() {
			var message_valid = $(this).attr('data-validation-dob');
		    $(this).rules('add', {
				validBirthDate: true,
				messages: {
					validBirthDate: message_valid
		        }
		    });
		});

		function populateDob() {
			var $container = $('.birthday-selects');
			var $day = $container.find('#DobDay');
			var $month = $container.find('#DobMonth');
			var $year = $container.find('#DobYear');
			// get values
			var $dayVal = $day.val().toString();
			var $monthVal = $month.val().toString();
			var $yearVal = $year.val().toString();

			$('.dob-hidden').val($yearVal + '/' + $monthVal + '/' + $dayVal);
		}

		if ($('.birthday-selects').length) {
			populateDob();
		}

		$('.birthday-selects select').each(function() {
			$(this).on('change', function() {
				var wrapper = $(this).closest('.select-wrapper');
				populateDob();

				$('.dob-hidden').focus();
				setTimeout(function(){
					$('.dob-hidden').blur();
					var customDd = wrapper.find('.dk-selected');
					if (customDd.length) {
						customDd.focus();
					} else {
						wrapper.find('select');
					}
				},50);
			});
		});

		// Allow empty fields
		$('[data-validation-allowEmpty]').each(function(){
			$(this).rules('add', {
				allowEmpty: true
			});
		});
	},

	init: function() {
		if ( $('form').length ) {
			validation.setMethods();
			validation.validateForms();
			validation.setRules();
		}
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

	// validation.js 
	validation.init();

	$('#nav-icon').click(function(){
		$(this).toggleClass('open');
	});
	$('.hamburber-menu').click(function(){
		$('.nav').toggleClass('mobile-hide').toggleClass('mobile-show');
		$('.about').toggleClass('mobile-hide').toggleClass('mobile-show');
	});
	$('.products-link, .solutions-link').click(function(){
		$('.subnav').toggleClass('mobile-hide').toggleClass('mobile-show');
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