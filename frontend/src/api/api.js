const BASE_URL = 'http://192.168.2.137:8000'

export async function ping() {
  const response = await fetch(`${BASE_URL}/ping`)
  return response.json()
}

export async function predict(csvFile) {
  const formData = new FormData()
  formData.append('file', csvFile)

  const response = await fetch(`${BASE_URL}/run/`, {
    method: 'POST',
    body: formData,
  })
  return response.json()
}
