$(function () {
    async function getTranscribedFile(file) {
        const formData = new FormData();
        formData.append('file', file);

        let response = await fetch('/transcribing', {
            method: 'POST',
            body: formData
        })
        if (response.ok) {
            let data = await response.json()
            return data['transcribed_text']
        }
        return ''
    }

    async function getSummarizedText(text, summarizationType) {
        let response = await fetch(`/${summarizationType}`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json;charset=utf-8'
            },
            body: JSON.stringify({
                'text': text
            })
        })
        if (response.ok) {
            let data = await response.json()
            return data['summarized_text']
        }
        return ''
    }

    async function sendSummarizedText(summarizedText, title) {
        let response = await fetch(`/history`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json;charset=utf-8'
            },
            body: JSON.stringify({
                'title': title,
                'text': summarizedText
            })
        })
    }

    let files = null;

    $('#login').on('click', function (event) {
        event.preventDefault()
        $('#login_modal').modal('show')
    })

    $('#reg').on('click', function (event) {
        event.preventDefault()
        $('#reg_modal').modal('show')
    })

    $('#dropArea').on({
        dragover: function (e) {
            e.preventDefault();
            $(this).addClass('dragover');
        },
        dragleave: function () {
            $(this).removeClass('dragover');
        },
        drop: function (e) {
            e.preventDefault();
            $(this).removeClass('dragover');
            files = e.originalEvent.dataTransfer.files;
            $('#get_annotation').prop('disabled', false)
        }
    });

    $('#get_annotation').on('click', async function (event) {
        event.preventDefault()
        if (files !== null) {
            let text = await getTranscribedFile(files[0])
            if (text !== '') {
                let summarizedText = await getSummarizedText(text, 'extractive')
                $('#annotation').val(summarizedText);
                $('#get_annotation').prop('disabled', true)
                await sendSummarizedText(summarizedText, files[0].name.replace(/\.[^/.]+$/, ""))
            }
        } else {
            console.log('Файлы не загружены');
        }
    })
});
