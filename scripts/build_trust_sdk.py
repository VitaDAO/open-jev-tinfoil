"""Reproducible opt-in SDK wheel patch; never alters the installed SDK."""
import argparse
import base64
import csv
import difflib
import hashlib
import io
from pathlib import Path
import zipfile

INPUT_SHA = 'f04e1c0ed98e22619c03e6fc8b9a2cab60a4e9ed2001ffec58ab5cd848d77642'
VERSION = '0.14.0+vita1'
SOURCE_HASHES = {
    'tinfoil/client.py':'aa41ea623554d714fa6c94ef5709943be10fd7475b6ccfcf5a9c0a9801654366',
    'tinfoil/sigstore.py':'8f36d7cf4ea9162451c7156afdfb653a2a1e01a0fcfec7f3b4971d01e8a48576',
}


def patched(name, source):
    if name == 'tinfoil/client.py':
        source = source.replace('user_cache_secret: Optional[str] = None):',
            'user_cache_secret: Optional[str] = None, sigstore_verifier_factory=None):', 1)
        source = source.replace('        self.enclave = enclave or ""',
            '        self._sigstore_verifier_factory = sigstore_verifier_factory\n'
            '        self.enclave = enclave or ""', 1)
        source = source.replace('sigstore_bundle, digest, self.repo, release.tag\n',
            'sigstore_bundle, digest, self.repo, release.tag,\n'
            '                    verifier=(self._sigstore_verifier_factory()\n'
            '                              if self._sigstore_verifier_factory is not None else None),\n', 1)
        source = source.replace('bundle.sigstore_bundle, bundle.digest, self.repo, bundle.release_tag\n',
            'bundle.sigstore_bundle, bundle.digest, self.repo, bundle.release_tag,\n'
            '                verifier=(self._sigstore_verifier_factory()\n'
            '                          if self._sigstore_verifier_factory is not None else None),\n', 1)
    elif name == 'tinfoil/sigstore.py':
        source = source.replace('expected_release_tag: Optional[str] = None\n',
            'expected_release_tag: Optional[str] = None, *, verifier: Optional[Verifier] = None\n')
        source = source.replace('    verifier = Verifier.production()',
            '    if verifier is None:\n        verifier = Verifier.production()', 1)
        source = source.replace('_verify_dsse_bundle(bundle_json, digest, repo, expected_release_tag)',
            '_verify_dsse_bundle(bundle_json, digest, repo, expected_release_tag, verifier=verifier)', 1)
    return source


def build(wheel, output, patch_output=None):
    raw = Path(wheel).read_bytes()
    if hashlib.sha256(raw).hexdigest() != INPUT_SHA:
        raise ValueError('Unexpected upstream wheel digest')
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        files = {name:archive.read(name) for name in archive.namelist()}
    diffs = []
    for name, digest in SOURCE_HASHES.items():
        if hashlib.sha256(files[name]).hexdigest() != digest:
            raise ValueError('Unexpected upstream source digest')
        original = files[name].decode()
        changed = patched(name, original)
        compile(changed, name, 'exec')
        files[name] = changed.encode()
        diffs.extend(difflib.unified_diff(original.splitlines(True), changed.splitlines(True),
                                        fromfile='a/'+name, tofile='b/'+name, n=0))
    old_info, new_info = 'tinfoil-0.14.0.dist-info/', f'tinfoil-{VERSION}.dist-info/'
    files = {name.replace(old_info,new_info):data for name,data in files.items() if not name.endswith('/RECORD')}
    metadata = new_info+'METADATA'
    files[metadata] = files[metadata].replace(b'Version: 0.14.0\n', f'Version: {VERSION}\n'.encode(), 1)
    record = io.StringIO(newline='')
    writer = csv.writer(record, lineterminator='\n')
    for name, data in sorted(files.items()):
        digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b'=').decode()
        writer.writerow([name,'sha256='+digest,len(data)])
    writer.writerow([new_info+'RECORD','',''])
    files[new_info+'RECORD'] = record.getvalue().encode()
    with zipfile.ZipFile(output,'x',compression=zipfile.ZIP_STORED) as archive:
        for name, data in sorted(files.items()):
            info = zipfile.ZipInfo(name, date_time=(1980,1,1,0,0,0))
            info.external_attr = 0o644 << 16
            archive.writestr(info, data)
    if patch_output:
        Path(patch_output).write_text(''.join(diffs))
    print(hashlib.sha256(Path(output).read_bytes()).hexdigest(), output)


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('wheel');parser.add_argument('output');parser.add_argument('--patch-output')
    args=parser.parse_args();build(args.wheel,args.output,args.patch_output)
