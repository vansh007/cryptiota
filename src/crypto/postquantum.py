"""
Post-Quantum Cryptography wrapper using liboqs
Kyber (KEM) + AES-GCM hybrid encryption
"""

import time
import os
from typing import Tuple, Dict
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend

# ---- Global liboqs availability check ----
try:
    import oqs
    GLOBAL_LIBOQS_AVAILABLE = True
except ImportError:
    GLOBAL_LIBOQS_AVAILABLE = False
    print("⚠️  liboqs not available - using simulation mode")


class PostQuantumCrypto:
    """Post-quantum encryption using Kyber + AES"""
    
    def __init__(self, kem_algorithm: str = "Kyber768"):
        self.kem_algorithm = kem_algorithm
        self.backend = default_backend()

        # Instance-level availability (IMPORTANT)
        self.liboqs_available = GLOBAL_LIBOQS_AVAILABLE
        self.kem = None
        
        if self.liboqs_available:
            try:
                self.kem = oqs.KeyEncapsulation(kem_algorithm)
            except Exception as e:
                print(f"⚠️  {kem_algorithm} not available in liboqs, falling back to simulation")
                self.liboqs_available = False
        
        # Simulation parameters (Kyber sizes)
        self.sim_params = {
            'Kyber512':  {'pk_size': 800,  'sk_size': 1632, 'ct_size': 768,  'ss_size': 32},
            'Kyber768':  {'pk_size': 1184, 'sk_size': 2400, 'ct_size': 1088, 'ss_size': 32},
            'Kyber1024': {'pk_size': 1568, 'sk_size': 3168, 'ct_size': 1568, 'ss_size': 32}
        }
        
        if kem_algorithm not in self.sim_params:
            raise ValueError(f"Unsupported KEM algorithm: {kem_algorithm}")
    
    def generate_keypair(self) -> Tuple:
        """Generate PQ keypair"""
        start_time = time.time()
        
        if self.liboqs_available:
            public_key = self.kem.generate_keypair()
            private_key = self.kem.export_secret_key()
        else:
            params = self.sim_params[self.kem_algorithm]
            public_key = os.urandom(params['pk_size'])
            private_key = os.urandom(params['sk_size'])
            time.sleep(0.002)  # Simulated Kyber delay
        
        keygen_time = time.time() - start_time
        return private_key, public_key, keygen_time
    
    def encrypt_data(self, data: bytes, public_key: bytes) -> Tuple[Dict, Dict]:
        """
        Hybrid PQ encryption:
        Kyber KEM → shared secret → AES-GCM
        """
        metrics = {}
        
        # --- KEM encapsulation ---
        start_time = time.time()
        if self.liboqs_available:
            ciphertext_kem, shared_secret = self.kem.encap_secret(public_key)
        else:
            params = self.sim_params[self.kem_algorithm]
            ciphertext_kem = os.urandom(params['ct_size'])
            shared_secret = os.urandom(params['ss_size'])
            time.sleep(0.001)
        
        metrics['kem_encap_time'] = time.time() - start_time
        
        # --- Symmetric encryption ---
        aes_key = shared_secret[:32]
        nonce = os.urandom(12)
        
        start_time = time.time()
        cipher = Cipher(
            algorithms.AES(aes_key),
            modes.GCM(nonce),
            backend=self.backend
        )
        encryptor = cipher.encryptor()
        ciphertext_data = encryptor.update(data) + encryptor.finalize()
        tag = encryptor.tag
        
        metrics['symmetric_enc_time'] = time.time() - start_time
        metrics['total_enc_time'] = metrics['kem_encap_time'] + metrics['symmetric_enc_time']
        
        encrypted_blob = {
            'ciphertext_data': ciphertext_data,
            'ciphertext_kem': ciphertext_kem,
            'nonce': nonce,
            'tag': tag
        }
        
        # --- Overhead calculation ---
        original_size = len(data)
        encrypted_size = (
            len(ciphertext_data) +
            len(ciphertext_kem) +
            len(nonce) +
            len(tag)
        )
        
        metrics['storage_overhead_bytes'] = encrypted_size - original_size
        metrics['storage_overhead_percent'] = (encrypted_size / original_size - 1) * 100
        
        return encrypted_blob, metrics
    
    def decrypt_data(
        self,
        encrypted_blob: Dict,
        private_key: bytes,
        public_key: bytes = None
    ) -> Tuple[bytes, Dict]:
        """Decrypt PQ encrypted data"""
        metrics = {}
        
        # --- KEM decapsulation ---
        start_time = time.time()
        if self.liboqs_available:
            kem_dec = oqs.KeyEncapsulation(self.kem_algorithm, private_key)
            shared_secret = kem_dec.decap_secret(encrypted_blob['ciphertext_kem'])
        else:
            params = self.sim_params[self.kem_algorithm]
            shared_secret = os.urandom(params['ss_size'])
            time.sleep(0.001)
        
        metrics['kem_decap_time'] = time.time() - start_time
        
        # --- Symmetric decryption ---
        aes_key = shared_secret[:32]
        
        start_time = time.time()
        cipher = Cipher(
            algorithms.AES(aes_key),
            modes.GCM(encrypted_blob['nonce'], encrypted_blob['tag']),
            backend=self.backend
        )
        decryptor = cipher.decryptor()
        plaintext = decryptor.update(encrypted_blob['ciphertext_data']) + decryptor.finalize()
        
        metrics['symmetric_dec_time'] = time.time() - start_time
        metrics['total_dec_time'] = metrics['kem_decap_time'] + metrics['symmetric_dec_time']
        
        return plaintext, metrics


def test_pq_crypto():
    """Test post-quantum encryption"""
    print("🧪 Testing Post-Quantum Cryptography...")
    if not GLOBAL_LIBOQS_AVAILABLE:
        print("⚠️  Running in SIMULATION mode (install liboqs for real PQ crypto)")
    
    test_data = b"IoT sensor data: temperature=25.3C" * 100
    
    for algo in ["Kyber512", "Kyber768", "Kyber1024"]:
        print(f"\n--- {algo} ---")
        crypto = PostQuantumCrypto(algo)
        
        private_key, public_key, keygen_time = crypto.generate_keypair()
        print(f"Key generation: {keygen_time*1000:.2f} ms")
        
        encrypted_blob, enc_metrics = crypto.encrypt_data(test_data, public_key)
        print(f"Encryption time: {enc_metrics['total_enc_time']*1000:.2f} ms")
        print(f"Storage overhead: {enc_metrics['storage_overhead_percent']:.2f}%")
        
        decrypted_data, dec_metrics = crypto.decrypt_data(encrypted_blob, private_key)
        print(f"Decryption time: {dec_metrics['total_dec_time']*1000:.2f} ms")
        
        assert decrypted_data == test_data
        print("✅ Test passed!")


if __name__ == "__main__":
    test_pq_crypto()
