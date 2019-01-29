var api_url = "/recsys/api/";
//var video_url = "https://youtu.be/";
//var video_url = "https://www.youtube.com/watch?v=";
var video_url = "https://www.youtube.com/embed/";


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
			var input_movies = data.input_movies_info;
			var rec_movies = data.recommendation_movies_info;
			
			if(data.message == "user history does not exist") {
				alert("user history does not exist.");
				return;
			}
			
			// 결과 초기화
			$("#input_movie_area").empty();
			$("#rec_movie_area").empty();
			
			// 결과 출력
			input_movie_append(input_movies);
			rec_movie_append(rec_movies);
			
			$('#result_area').show();
		},
		error: function(jqXHR, textStatus, errorThrown) {
			alert(errorThrown);
		}
	});
});

// 시청 컨텐츠 추가
var input_movie_append = function(input_movies) {
	$.each( input_movies, function( i, val ) {
		console.log(val.movieId + " : " + val.title + " : " + val.url);
		var html = '<div class="col-sm-6 col-md-3" style="cursor:pointer" id="card" data-url="' + val.url + '">';
			html += '<div class="card card-accent-danger">';
			html += '<div class="card-body"><img width="100%" height="100%" src="' + 'https://i.ytimg.com/vi/' + val.url + '/hqdefault.jpg' + '"></div>';
			html += '<div class="card-footer">' + val.title + '</div>';
			html += '</div>';
			html += '</div>';
			
		$("#input_movie_area").append(html);
	});
}

// 추천 컨텐츠 추가
var rec_movie_append = function(rec_movies) {
	$.each( rec_movies, function( i, val ) {
		console.log(val.movieId + " : " + val.title + " : " + val.url);
		var html = '<div class="col-sm-6 col-md-3" style="cursor:pointer" id="card">';
			html += '<div class="card card-accent-primary">';
			html += '<div class="card-body"><img width="100%" height="100%" src="' + 'https://i.ytimg.com/vi/' + val.url + '/hqdefault.jpg' + '"></div>';
			html += ' <div class="card-footer">' + val.title + '</div>';
			html += '</div>';
			html += '</div>';
			
		$("#rec_movie_area").append(html);
	});
}

// card 클릭시 modalview 생성 이벤트
$(document).on('click', 'div[id="card"]', function(){
	// modal 초기화
	$('.modal-body').empty();
	// 영상 가져오기
	var html = '<iframe class="video" src="' + video_url + $(this).attr("data-url") + '" controls=0" autohide=1 frameborder="0"></iframe>';
	
	$('.modal-body').append(html);
	$('#videoModal').modal('show');
});

$(document).ready(function() {
	// 목록 불러오기
	
});

