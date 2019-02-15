var api_url = "/recsys/api/";

// 추천 시스템 API
$('#send').click(function() {
	if (typeof($("#user_id").val()) === "undefined" || $("#user_id").val() === "") {
		alert("user_id를 입력하세요.");
		return;
	}
	
	var _data = JSON.stringify({
			"userId": $("#user_id").val()
		});

	$.ajax({
		url: api_url,
		type: "POST",
		dataType:"json",
		contentType: "application/json; charset=utf-8",
		xhrFields: {
			withCredentials: false
		},
		data: _data,
		success: function(data) {
			//console.log(data);
			var item_history = data.history_items_info;
			var rec_items = data.recommendation_items_info;
			
			if(data.message == "user history does not exist") {
				alert("user history does not exist.");
				return;
			}
			
			// 결과 초기화
			$("#history_items_area").empty();
			$("#rec_items_area").empty();
			
			// 결과 출력
			item_history_append(item_history);
			rec_item_append(rec_items);
			
			$('#result_area').show();
		},
		error: function(jqXHR, textStatus, errorThrown) {
			alert(errorThrown);
		}
	});
});

// 시청 컨텐츠 추가
var item_history_append = function(input_movies) {
	$.each( input_movies, function( i, val ) {
		console.log(val.itemName + " : " + val.item_index + " : " + val.itemId);
		
		var html = '<div class="col-sm-6 col-md-3" style="cursor:pointer" id="card">';
			html += '<div class="card card-accent-danger">';
			html += '<div class="card-body">' + val.itemName + '</div>';
			html += '<div class="card-footer">' + val.itemId + '</div>';
			html += '</div>';
			html += '</div>';
			
		$("#history_items_area").append(html);
	});
}

// 추천 컨텐츠 추가
var rec_item_append = function(rec_movies) {
	$.each( rec_movies, function( i, val ) {
		console.log(val.itemName + " : " + val.item_index + " : " + val.itemId);
		
		var html = '<div class="col-sm-6 col-md-3" style="cursor:pointer" id="card">';
			html += '<div class="card card-accent-danger">';
			html += '<div class="card-body">' + val.itemName + '</div>';
			html += '<div class="card-footer">' + val.itemId + '</div>';
			html += '</div>';
			html += '</div>';
			
		$("#rec_items_area").append(html);
	});
}

// card 클릭시 modalview 생성 이벤트
// $(document).on('click', 'div[id="card"]', function(){
// 	// modal 초기화
// 	$('.modal-body').empty();
// 	// 영상 가져오기
// 	var html = '<iframe class="video" src="' + video_url + $(this).attr("data-url") + '" controls=0" autohide=1 frameborder="0"></iframe>';
	
// 	$('.modal-body').append(html);
// 	$('#videoModal').modal('show');
// });

$(document).ready(function() {
	// 목록 불러오기
	
});

