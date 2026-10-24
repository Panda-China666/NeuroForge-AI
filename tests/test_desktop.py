from types import SimpleNamespace
import pytest
from fastapi.testclient import TestClient
from neuroforge.app import app
from neuroforge import desktop_runtime, worker

client = TestClient(app)


def test_desktop_identity_and_owner_authentication(monkeypatch):
    assert client.get('/api/desktop/identity').json()['application'] == 'NeuroForge AI'
    monkeypatch.setenv('NEUROFORGE_DESKTOP_TOKEN', 'owner-secret')
    assert client.post('/api/desktop/stop').status_code == 403
    assert client.post('/api/desktop/stop', headers={'X-NeuroForge-Desktop': 'wrong'}).status_code == 403
    monkeypatch.setattr(desktop_runtime, 'SERVER', None)
    assert client.post('/api/desktop/stop', headers={'X-NeuroForge-Desktop': 'owner-secret'}).status_code == 409


def test_desktop_shutdown_preserves_active_jobs_and_quiesces_submissions(monkeypatch):
    monkeypatch.setenv('NEUROFORGE_DESKTOP_TOKEN', 'owner-secret')
    server = SimpleNamespace(should_exit=False)
    monkeypatch.setattr(desktop_runtime, 'SERVER', server)
    monkeypatch.setattr(worker, 'ACCEPTING_JOBS', True)
    monkeypatch.setattr(desktop_runtime.store, 'all_of', lambda kind: [{'id': 'active', 'status': 'running'}])
    headers = {'X-NeuroForge-Desktop': 'owner-secret'}
    assert client.post('/api/desktop/stop', headers=headers).status_code == 409
    assert worker.ACCEPTING_JOBS and not server.should_exit
    monkeypatch.setattr(desktop_runtime.store, 'all_of', lambda kind: [])
    callbacks = []
    class Timer:
        def __init__(self, delay, callback): callbacks.append(callback)
        def start(self): pass
    monkeypatch.setattr(desktop_runtime.threading, 'Timer', Timer)
    assert client.post('/api/desktop/stop', headers=headers).status_code == 200
    assert not worker.ACCEPTING_JOBS and not server.should_exit
    with pytest.raises(ValueError, match='Runtime is stopping'):
        worker.submit('train', {})
    callbacks[0]()
    assert server.should_exit


def test_custom_desktop_port_origin(monkeypatch):
    monkeypatch.setenv('NEUROFORGE_PORT', '8877')
    response = client.post('/api/analyze', json={'ir': {}}, headers={'Origin': 'http://127.0.0.1:8877'})
    assert response.status_code != 403
    assert client.post('/api/analyze', json={'ir': {}}, headers={'Origin': 'http://127.0.0.1:8876'}).status_code == 403
